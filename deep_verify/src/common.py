# coding=utf-8
# Copyright 2019 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Graph construction for dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator

import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


def with_explicit_update(volatile_value):
  """Wraps the volatile value in a variable to cache a stationary copy of it.

  Args:
    volatile_value: Volatile tensor value to hold still; may be nested.

  Returns:
    materialised_value: Non-trainable variable (or nest thereof) to hold a
      stationary copy of the tensor.
    update_op: Operation to update the cache by reevaluating the tensor.
  """
  def materialise(value):
    """Returns a non-trainable variable to shadow the given volatile tensor."""
    return tf.get_variable(value.name.replace(':', '__') + '_materialised',
                           shape=value.shape,
                           dtype=value.dtype,
                           trainable=False)

  nest = tf.contrib.framework.nest
  materialised_value = nest.map_structure(materialise, volatile_value)
  update_op = tf.group(nest.flatten(
      nest.map_structure(tf.assign, materialised_value, volatile_value)))
  return materialised_value, update_op


def targeted_objective(final_w, final_b, labels):
  """Determines final layer weights for attacks targeting each class.

  Args:
    final_w: 2D tensor of shape (last_hidden_layer_size, num_classes)
      containing the weights for the final linear layer.
    final_b: 1D tensor of shape (num_classes) containing the biases for the
      final hidden layer.
    labels: 1D integer tensor of shape (batch_size) of labels for each
      input example.

  Returns:
    obj_w: Tensor of shape (num_classes, batch_size, last_hidden_layer_size)
      containing weights (to use in place of final linear layer weights)
      for targeted attacks.
    obj_b: Tensor of shape (num_classes, batch_size) containing bias
      (to use in place of final linear layer biases) for targeted attacks.
  """
  # Elide objective with final linear layer.
  final_wt = tf.transpose(final_w)
  obj_w = tf.expand_dims(final_wt, axis=1) - tf.gather(final_wt, labels, axis=0)
  obj_b = tf.expand_dims(final_b, axis=1) - tf.gather(final_b, labels, axis=0)
  return obj_w, obj_b


def concave_max_binsearch(fn, lb, ub, num_iter=20):
  """Ternary search to find the maximum of the given concave function.

  Although the branching factor is three, the search interval shrinks by a
  factor of two each iteration. Therefore, 20 iterations (the default) give
  an accuracy of around one part per million.

  The returned tensor will be a tuple of `(argmax, max)`. `argmax` has no
  gradients. `max` is the tensor returned by applying `fn` to argmax, and so
  will have no gradients via `argmax` but may have gradients with respect to
  other tensors captured by `fn`.

  Args:
    fn: Function accepting a tensor and returning a tensor of the same shape,
      expressing concave functions applied element-wise. Each output element
      should depend only upon the corresponding input element.
    lb: Floating-point tensor containing lower-bounds on inputs to `fn`.
    ub: Floating-point tensor containing upper-bounds on inputs to `fn`.
    num_iter: Number of binary search iterations to perform.

  Returns:
    Pair of tensors (of same shape as lb and ub) containing:
      argmax: inputs (in range lb...ub) that maximise `fn` element-wise.
      max: the attained values of `fn`.
  """
  mid = tf.stop_gradient(.5 * lb + .5 * ub)
  f_mid = fn(mid)

  for _ in range(num_iter):
    # Calculate quartiles.
    lq = tf.stop_gradient(.75 * lb + .25 * ub)
    uq = tf.stop_gradient(.25 * lb + .75 * ub)
    f_lq = fn(lq)
    f_uq = fn(uq)

    # Identify three cases, recalling that fn is concave.
    # Case 1: f_lq > f_mid > f_uq
    # The maximum occurs in the range [lb, mid].
    # Case 2: f_lq > f_mid > f_uq
    # The maximum occurs in the range [mid, ub].
    # Case 3: f_lq < f_mid > f_uq
    # The maximum occurs in the range [lq, uq].
    case1 = f_lq > f_mid
    case2 = f_uq > f_mid
    lb, ub, mid, f_mid = (
        tf.where(case1, lb, tf.where(case2, mid, lq)),
        tf.where(case1, mid, tf.where(case2, ub, uq)),
        tf.where(case1, lq, tf.where(case2, uq, mid)),
        tf.where(case1, f_lq, tf.where(case2, f_uq, f_mid))
    )

  return mid, f_mid


def _prod(lst):
  return functools.reduce(operator.mul, lst, 1)


def conv_output_shape(input_shape, w, padding, strides):
  """Calculates the output shape of the given N-D convolution.

  Args:
    input_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the inputs: [input_height, input_width, input_channels].
    w: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    Integer list of length N+1 specifying the non-batch dimensions
      of the outputs: [output_height, output_width, output_channels].

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  # Connect a convolution (never to be run) to infer the output's
  # spatial structure.
  dummy_inputs = tf.zeros(dtype=w.dtype, shape=([1] + input_shape))
  if len(w.shape) == 4:
    dummy_outputs = tf.nn.convolution(dummy_inputs,
                                      w, padding=padding, strides=strides)
  elif len(w.shape) == 3:
    dummy_outputs = tf.nn.conv1d(dummy_inputs,
                                 w, padding=padding, stride=strides[0])
  else:
    raise ValueError()
  return dummy_outputs.shape.as_list()[1:]


def convolution(x, kernel, padding, strides):
  """Applies an N-D convolution, respecting two batch dimensions.

  Args:
    x: (N+3)D tensor of shape (num_classes, batch_size, input_height,
      input_width, input_channels) containing the coefficients to which the
      convolution is to be applied.
    kernel: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    (N+3)D tensor of shape (num_classes, batch_size, output_height,
      output_width, output_channels) containing the convolution of `x`.

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  # Temporarily combine the classes/batch dimensions while convolving.
  num_classes = x.shape[0].value
  batch_size = tf.shape(x)[1]
  x_squeezed = tf.reshape(x, shape=([num_classes * batch_size] +
                                    x.shape[2:].as_list()))
  if len(kernel.shape) == 4:
    y = tf.nn.convolution(x_squeezed, kernel, padding=padding, strides=strides)
  elif len(kernel.shape) == 3:
    y = tf.nn.conv1d(x_squeezed, kernel, padding=padding, stride=strides[0])
  else:
    raise ValueError()
  return tf.reshape(y, shape=([num_classes, batch_size] +
                              y.shape[1:].as_list()))


def conv_transpose(y, kernel, result_shape, padding, strides):
  """Applies an N-D transpose-convolution, respecting two batch dimensions.

  Args:
    y: (N+3)D tensor of shape (num_classes, batch_size, output_height,
      output_width, output_channels) containing the coefficients to which the
      transpose-convolution is to be applied.
    kernel: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    result_shape: List of [input_height, input_width, input_channels] specifying
      the N+1 trailing (non-batch) dimensions of the output.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    (N+3)D tensor of shape (num_classes, batch_size, input_height,
      input_width, input_channels) containing the transpose-convolution of `y`.

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  # Temporarily combine the (num_classes, batch_size) dimensions
  # while applying the transpose convolution.
  batch_size = tf.shape(y)[1]
  y_squeezed = tf.reshape(y,
                          shape=([y.shape[0].value * batch_size] +
                                 y.shape[2:].as_list()))

  if len(result_shape) == 3:
    x = tf.nn.conv2d_transpose(
        y_squeezed, kernel,
        output_shape=([tf.shape(y_squeezed)[0]] + result_shape),
        padding=padding, strides=([1] + list(strides) + [1]))
  elif len(result_shape) == 2:
    x = tf.contrib.nn.conv1d_transpose(
        y_squeezed, kernel,
        output_shape=([tf.shape(y_squeezed)[0]] + result_shape),
        padding=padding, strides=strides[0])
  else:
    raise ValueError()
  return tf.reshape(x, shape=(
      [y.shape[0].value, batch_size] + x.shape[1:].as_list()))


def avgpool_transpose(y, result_shape, kernel_shape, strides):
  """Applies an N-D 'transposed average pool', respecting two batch dimensions.

  Args:
    y: (N+3)D tensor of shape (num_classes, batch_size, output_height,
      output_width, channels) containing the coefficients to which the
      transpose-convolution is to be applied.
    result_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the result: [input_height, input_width, channels].
    kernel_shape: Integer list of `[kernel_height, kernel_width]`,
      or `None` to aggregate over the layer`s entire spatial extent.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    (N+3)D tensor of shape (num_classes, batch_size, input_height,
      input_width, channels) containing the transpose-avgpool of `y`.
  """
  if kernel_shape is None:
    # We know that output_height=1 and output_width=1.
    return tf.tile(y, [1, 1] + list(result_shape[:-1]) + [1]) / (
        _prod(result_shape[:-1]))

  else:
    # Treat the average pool as a convolution with uniform weights.
    kernel = tf.ones(dtype=y.dtype, shape=(list(kernel_shape) + [1, 1]))
    channels = result_shape[-1]
    kernel *= tf.eye(channels, dtype=y.dtype)
    kernel /= _prod(kernel_shape)
    return conv_transpose(y, kernel, result_shape=result_shape,
                          padding='VALID', strides=strides)


def conv_broadcast(x, kernel_shape, padding, strides):
  """Performs an N-D convolutional broadcast.

  Inserts dimensions into `x`, by duplicating elements with respect to the
  specified convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to each spatial dimension of the convolution (and disregarding
  batch and channel dimensions for the sake of illustration), input data
  of the form [a, b, c, d, e, f] is mapped to::
      [[ a, b, c ],
       [ b, c, d ],
       [ c, d, e ],
       [ d, e, f ]]

  The output size (height and width) is as determined by tf.nn.convolution
  with the kernel size, stride, and padding specified.

  Args:
    x: (N+2)D tensor of shape (batch_size, input_height, input_width,
      input_channels)
    kernel_shape: Integer list of length N specifying the spatial shape of the
      convolution kernel.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    (2N+3)D tensor of shape (batch_size,
    output_height, output_width, 1,
    kernel_height, kernel_width, input_channels).

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  if len(kernel_shape) == 2:
    return conv2d_broadcast(x, kernel_shape[0], kernel_shape[1],
                            padding, strides)
  elif len(kernel_shape) == 1:
    return conv1d_broadcast(x, kernel_shape[0], padding, strides[0])
  else:
    raise ValueError()


def conv2d_broadcast(x, kernel_height, kernel_width, padding, strides):
  """Performs a convolutional broadcast.

  Inserts dimensions into `x`, by duplicating elements with respect to the
  specified convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to each spatial dimension of the convolution (and disregarding
  batch and channel dimensions for the sake of illustration), input data
  of the form [a, b, c, d, e, f] is mapped to::
      [[ a, b, c ],
       [ b, c, d ],
       [ c, d, e ],
       [ d, e, f ]]

  The output size (height and width) is as determined by tf.nn.convolution
  with the kernel size, stride, and padding specified.

  Args:
    x: 4D tensor of shape (batch_size, input_height, input_width,
      input_channels)
    kernel_height: Height of the convolution kernel.
    kernel_width: Width of the convolution kernel.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    7D tensor of shape (batch_size,
    output_height, output_width, 1,
    kernel_height, kernel_width, input_channels).
  """
  batch_size = tf.shape(x)[0]
  input_channels = x.shape[3].value

  # Temporarily combine the (batch_size, input_channels) dims while
  # applying the convolution. Introduce a dummy channels dimension instead.
  squeezed = tf.transpose(x, perm=[0, 3, 1, 2])
  squeezed = tf.reshape(squeezed, shape=(
      [batch_size * input_channels] +
      x.shape[1:3].as_list() +
      [1]))

  # Convolve each elementary (i.e. one-hot) filter with x.
  diagonal_kernel = tf.reshape(
      tf.eye(kernel_height * kernel_width, dtype=x.dtype),
      shape=[kernel_height, kernel_width, 1, kernel_height * kernel_width])
  conv = tf.nn.convolution(
      squeezed, diagonal_kernel,
      padding=padding, strides=strides)

  # The resulting convolution has shape (batch_size*input_channels,
  # output_height, output_width, kernel_height*kernel_width).
  # Move input_channels back to the last dimension.
  result = tf.reshape(conv, shape=(
      [batch_size, input_channels] +
      conv.shape[1:3].as_list() +
      [kernel_height, kernel_width]))
  result = tf.transpose(result, perm=[0, 2, 3, 4, 5, 1])

  # Insert output_channels dimension.
  return tf.expand_dims(result, 3)


def conv1d_broadcast(x, kernel_length, padding, stride):
  """Performs a convolutional broadcast.

  Inserts dimensions into `x`, by duplicating elements with respect to the
  specified convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to the sequence dimension of the convolution (and disregarding
  batch and channel dimensions for the sake of illustration), input data
  of the form [a, b, c, d, e, f] is mapped to::
      [[ a, b, c ],
       [ b, c, d ],
       [ c, d, e ],
       [ d, e, f ]]

  The output size (length) is as determined by tf.nn.convolution
  with the kernel size, stride, and padding specified.

  Args:
    x: 3D tensor of shape (batch_size, input_length, input_channels)
    kernel_length: Length of the convolution kernel.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    stride: Integer stride.

  Returns:
    5D tensor of shape (batch_size,
    output_length, 1,
    kernel_length, input_channels).
  """
  batch_size = tf.shape(x)[0]
  input_channels = x.shape[2].value

  # Temporarily combine the (batch_size, input_channels) dims while
  # applying the convolution. Introduce a dummy channels dimension instead.
  squeezed = tf.transpose(x, perm=[0, 2, 1])
  squeezed = tf.reshape(squeezed, shape=(
      [batch_size * input_channels] +
      x.shape[1:2].as_list() +
      [1]))

  # Convolve each elementary (i.e. one-hot) filter with x.
  diagonal_kernel = tf.reshape(
      tf.eye(kernel_length, dtype=x.dtype),
      shape=[kernel_length, 1, kernel_length])
  conv = tf.nn.conv1d(
      squeezed, diagonal_kernel,
      padding=padding, stride=stride)

  # The resulting convolution has shape (batch_size*input_channels,
  # output_length, kernel_length).
  # Move input_channels back to the last dimension.
  result = tf.reshape(conv, shape=(
      [batch_size, input_channels] +
      conv.shape[1:2].as_list() +
      [kernel_length]))
  result = tf.transpose(result, perm=[0, 2, 3, 1])

  # Insert output_channels dimension.
  return tf.expand_dims(result, 2)


def conv_reduce_sum(x, result_shape, padding, strides):
  """Sums along the output dimensions in line with an N-D convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to each spatial dimension of the convolution (and disregarding
  class, batch and channel dimensions for the sake of illustration), input data
  of the form::
      [[ a, b, c ],
       [ d, e, f ],
       [ g, h, i ],
       [ j, k, l ]]
  is mapped to [a, b+d, c+e+g, f+h+j, i+k, l].

  Args:
    x: (2N+4)D tensor of shape (num_classes, batch_size,
      output_height, output_width, output_channels,
      kernel_height, kernel_width, input_channels).
    result_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the result: [input_height, input_width, input_channels].
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    (N+3)D tensor of shape (num_classes, batch_size,
      input_height, input_width, input_channels)

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  if len(result_shape) == 3:
    return conv2d_reduce_sum(x, result_shape[0], result_shape[1],
                             padding, strides)
  elif len(result_shape) == 2:
    return conv1d_reduce_sum(x, result_shape[0], padding, strides[0])
  else:
    raise ValueError()


def conv2d_reduce_sum(x, input_height, input_width, padding, strides):
  """Sums along the output dimensions in line with a convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to each spatial dimension of the convolution (and disregarding
  class, batch and channel dimensions for the sake of illustration), input data
  of the form::
      [[ a, b, c ],
       [ d, e, f ],
       [ g, h, i ],
       [ j, k, l ]]
  is mapped to [a, b+d, c+e+g, f+h+j, i+k, l].

  Args:
    x: 8D tensor of shape (num_classes, batch_size,
      output_height, output_width, output_channels,
      kernel_height, kernel_width, input_channels).
    input_height: height of the returned tensor.
    input_width: width of the returned tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    5D tensor of shape (num_classes, batch_size,
      input_height, input_width, input_channels)
  """
  # Sum over the output channels.
  lam_sum = tf.reduce_sum(x, axis=4)

  num_classes = x.shape[0].value
  batch_size = tf.shape(x)[1]
  kernel_height = x.shape[5].value
  kernel_width = x.shape[6].value
  input_channels = x.shape[7].value

  # Temporarily combine the (num_classes, batch_size, in_layer_channels) dims
  # while applying a transpose convolution.
  # Also combine (kernel_height, kernel_width), using them as the channels
  # as we'll apply the transpose convolution to each kernel point separately.
  lam_squeezed = tf.transpose(lam_sum, perm=[0, 1, 6, 2, 3, 4, 5])
  lam_squeezed = tf.reshape(lam_squeezed, shape=(
      [num_classes * batch_size * input_channels] +
      x.shape[2:4].as_list() +
      [kernel_height * kernel_width]))

  # De-convolve each elementary (i.e. one-hot) filter with the corresponding
  # slice of lambda.
  diagonal_kernel = tf.reshape(
      tf.eye(kernel_height * kernel_width, dtype=x.dtype),
      shape=[kernel_height, kernel_width, 1, kernel_height * kernel_width])
  lam_deconv = tf.nn.conv2d_transpose(
      lam_squeezed, diagonal_kernel, output_shape=(
          [num_classes * batch_size * input_channels] +
          [input_height, input_width, 1]),
      padding=padding, strides=([1] + list(strides) + [1]))

  # The resulting de-convolution has shape
  # (num_classes*batch_size*in_layer_channels,
  # in_layer_height, in_layer_width, 1).
  # Make it match mu_in.
  result = tf.reshape(lam_deconv, shape=(
      [num_classes, batch_size, input_channels] +
      lam_deconv.shape[1:3].as_list()))
  return tf.transpose(result, perm=[0, 1, 3, 4, 2])


def conv1d_reduce_sum(x, input_length, padding, stride):
  """Sums along the output dimensions in line with a convolution.

  For example, with a kernel size of 3, padding=`VALID`, and stride of 1, then
  with respect to the spatial dimension of the convolution (and disregarding
  class, batch and channel dimensions for the sake of illustration), input data
  of the form::
      [[ a, b, c ],
       [ d, e, f ],
       [ g, h, i ],
       [ j, k, l ]]
  is mapped to [a, b+d, c+e+g, f+h+j, i+k, l].

  Args:
    x: 6D tensor of shape (num_classes, batch_size,
      output_length, output_channels,
      kernel_length, input_channels).
    input_length: length of the returned tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    stride: Integer stride.

  Returns:
    4D tensor of shape (num_classes, batch_size,
      input_length, input_channels)
  """
  # Sum over the output channels.
  lam_sum = tf.reduce_sum(x, axis=3)

  num_classes = x.shape[0].value
  batch_size = tf.shape(x)[1]
  kernel_length = x.shape[4].value
  input_channels = x.shape[5].value

  # Temporarily combine the (num_classes, batch_size, in_layer_channels) dims
  # while applying a transpose convolution.
  # Also use (kernel_length) as the channels
  # as we'll apply the transpose convolution to each kernel point separately.
  lam_squeezed = tf.transpose(lam_sum, perm=[0, 1, 4, 2, 3])
  lam_squeezed = tf.reshape(lam_squeezed, shape=(
      [num_classes * batch_size * input_channels] +
      x.shape[2:3].as_list() +
      [kernel_length]))

  # De-convolve each elementary (i.e. one-hot) filter with the corresponding
  # slice of lambda.
  diagonal_kernel = tf.reshape(
      tf.eye(kernel_length, dtype=x.dtype),
      shape=[kernel_length, 1, kernel_length])
  lam_deconv = tf.contrib.nn.conv1d_transpose(
      lam_squeezed, diagonal_kernel, output_shape=(
          [num_classes * batch_size * input_channels] +
          [input_length, 1]),
      padding=padding, strides=stride)

  # The resulting de-convolution has shape
  # (num_classes*batch_size*in_layer_channels,
  # in_layer_length, 1).
  # Make it match mu_in.
  result = tf.reshape(lam_deconv, shape=(
      [num_classes, batch_size, input_channels] +
      lam_deconv.shape[1:2].as_list()))
  return tf.transpose(result, perm=[0, 1, 3, 2])


def materialise_conv(w, b, input_shape, padding, strides):
  """Converts an N-D convolution to an equivalent linear layer.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the inputs: [input_height, input_width, input_channels].
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    w: 2D tensor of shape (input_height * input_width * input_channels,
      output_height * output_width * output_channels) containing weights.
    b: 1D tensor of shape (output_height * output_width * output_channels)
      containing biases, or `None` if no biases.

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  if len(input_shape) == 3:
    return materialise_conv2d(w, b, input_shape[0], input_shape[1],
                              padding, strides)
  elif len(input_shape) == 2:
    return materialise_conv1d(w, b, input_shape[0], padding, strides[0])
  else:
    raise ValueError()


def materialise_conv2d(w, b, input_height, input_width, padding, strides):
  """Converts a convolution to an equivalent linear layer.

  Args:
    w: 4D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_height: height of the input tensor.
    input_width: width of the input tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    w: 2D tensor of shape (input_height * input_width * input_channels,
      output_height * output_width * output_channels) containing weights.
    b: 1D tensor of shape (output_height * output_width * output_channels)
      containing biases, or `None` if no biases.
  """
  kernel_height = w.shape[0].value
  kernel_width = w.shape[1].value
  input_channels = w.shape[2].value
  output_channels = w.shape[3].value

  # Temporarily move the input_channels dimension to output_channels.
  w = tf.reshape(w, shape=(kernel_height, kernel_width, 1,
                           input_channels * output_channels))
  # Apply the convolution to elementary (i.e. one-hot) inputs.
  diagonal_input = tf.reshape(
      tf.eye(input_height * input_width, dtype=w.dtype),
      shape=[input_height * input_width, input_height, input_width, 1])
  conv = tf.nn.convolution(
      diagonal_input, w,
      padding=padding, strides=strides)
  output_height = conv.shape[1].value
  output_width = conv.shape[2].value
  # conv is of shape (input_height * input_width, output_height, output_width,
  #                   input_channels * output_channels).
  # Reshape it to (input_height * input_width * input_channels,
  #                output_height * output_width * output_channels).
  w = tf.reshape(conv, shape=(
      [input_height * input_width,
       output_height, output_width,
       input_channels, output_channels]))
  w = tf.transpose(w, perm=[0, 3, 1, 2, 4])
  w = tf.reshape(w, shape=(
      [input_height * input_width * input_channels,
       output_height * output_width * output_channels]))

  # Broadcast b over spatial dimensions.
  b = tf.tile(b, [output_height * output_width]) if b is not None else None

  return w, b


def materialise_conv1d(w, b, input_length, padding, stride):
  """Converts a convolution to an equivalent linear layer.

  Args:
    w: 3D tensor of shape (kernel_length, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_length: length of the input tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    stride: Integer stride.

  Returns:
    w: 2D tensor of shape (input_length * input_channels,
      output_length * output_channels) containing weights.
    b: 1D tensor of shape (output_length * output_channels)
      containing biases, or `None` if no biases.
  """
  kernel_length = w.shape[0].value
  input_channels = w.shape[1].value
  output_channels = w.shape[2].value

  # Temporarily move the input_channels dimension to output_channels.
  w = tf.reshape(w, shape=(kernel_length, 1,
                           input_channels * output_channels))
  # Apply the convolution to elementary (i.e. one-hot) inputs.
  diagonal_input = tf.reshape(
      tf.eye(input_length, dtype=w.dtype),
      shape=[input_length, input_length, 1])
  conv = tf.nn.conv1d(
      diagonal_input, w,
      padding=padding, stride=stride)
  output_length = conv.shape[1].value
  # conv is of shape (input_length, output_length,
  #                   input_channels * output_channels).
  # Reshape it to (input_length * input_channels,
  #                output_length * output_channels).
  w = tf.reshape(conv, shape=(
      [input_length,
       output_length,
       input_channels, output_channels]))
  w = tf.transpose(w, perm=[0, 2, 1, 3])
  w = tf.reshape(w, shape=(
      [input_length * input_channels,
       output_length * output_channels]))

  # Broadcast b over spatial dimensions.
  b = tf.tile(b, [output_length]) if b is not None else None

  return w, b


def decode_batchnorm(batchnorm_module):
  """Calculates the neuron-wise multipliers and biases of the batch norm layer.

  Note that, in the case of a convolution, the returned bias will have
  spatial dimensions.

  Args:
    batchnorm_module: `snt.BatchNorm` module.

  Returns:
    w: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      neuron-wise multipliers for the batch norm layer.
    b: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      neuron-wise biases for the batch norm layer.
  """
  if isinstance(batchnorm_module, ibp.BatchNorm):
    mean = batchnorm_module.mean
    variance = batchnorm_module.variance
    variance_epsilon = batchnorm_module.epsilon
    scale = batchnorm_module.scale
    offset = batchnorm_module.bias

  else:
    assert isinstance(batchnorm_module, snt.BatchNorm)
    mean = batchnorm_module.moving_mean
    variance = batchnorm_module.moving_variance
    variance_epsilon = batchnorm_module._eps  # pylint: disable=protected-access
    try:
      scale = batchnorm_module.gamma
    except snt.Error:
      scale = None
    try:
      offset = batchnorm_module.beta
    except snt.Error:
      offset = None

  w = tf.rsqrt(variance + variance_epsilon)
  if scale is not None:
    w *= scale

  b = -w * mean
  if offset is not None:
    b += offset

  # Batchnorm vars have a redundant leading dim.
  w = tf.squeeze(w, axis=0)
  b = tf.squeeze(b, axis=0)
  return w, b


def combine_with_batchnorm(w, b, batchnorm_module):
  """Combines a linear layer and a batch norm into a single linear layer.

  Calculates the weights and biases of the linear layer formed by
  applying the specified linear layer followed by the batch norm.

  Note that, in the case of a convolution, the returned bias will have
  spatial dimensions.

  Args:
    w: 2D tensor of shape (input_size, output_size) or 4D tensor of shape
      (kernel_height, kernel_width, input_channels, output_channels) containing
      weights for the linear layer.
    b: 1D tensor of shape (output_size) or (output_channels) containing biases
      for the linear layer, or `None` if no bias.
    batchnorm_module: `snt.BatchNorm` module.

  Returns:
    w: 2D tensor of shape (input_size, output_size) or 4D tensor of shape
      (kernel_height, kernel_width, input_channels, output_channels) containing
      weights for the combined layer.
    b: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      biases for the combined layer.
  """
  if b is None:
    b = tf.zeros(dtype=w.dtype, shape=())

  w_bn, b_bn = decode_batchnorm(batchnorm_module)
  return w * w_bn, b * w_bn + b_bn
