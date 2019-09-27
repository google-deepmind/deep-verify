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

"""Naive bound calculation for common neural network layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src import common
from deep_verify.src.bounds import layer_bounds
import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


class IntervalBounds(ibp.AbstractBounds):
  """Upper and lower bounds, as a delta relative to nominal values."""

  def __init__(self, lower_rel, upper_rel, nominal):
    super(IntervalBounds, self).__init__()
    self._lower_rel = lower_rel
    self._upper_rel = upper_rel
    self._nominal = nominal
    self._update_cached_bounds_op = None

  @property
  def lower_rel(self):
    """Returns lower bounds, expressed relative to nominal values."""
    return self._lower_rel

  @property
  def upper_rel(self):
    """Returns upper bounds, expressed relative to nominal values."""
    return self._upper_rel

  @property
  def nominal(self):
    return self._nominal

  @property
  def lower(self):
    """Returns absolute lower bounds."""
    return self.nominal + self.lower_rel

  @property
  def upper(self):
    """Returns absolute upper bounds."""
    return self.nominal + self.upper_rel

  @property
  def shape(self):
    return self.lower_rel.shape.as_list()

  def apply_batch_reshape(self, wrapper, shape):
    """Propagates the bounds through a reshape.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      shape: output shape, excluding the batch dimension.

    Returns:
      Output bounds.
    """
    reshape = snt.BatchReshape(shape)
    return IntervalBounds(reshape(self.lower_rel), reshape(self.upper_rel),
                          reshape(self.nominal))

  def apply_sequence_average(self, denom_for_avg):
    """Propagates the bounds through a sequence average layer.

    Args:
      denom_for_avg: Divisor to apply after a `reduce_sum(x, axis=1)` operation.
        For inputs of shape (batch_size, max_sequence_length, input_channels),
        this will be a 2D tensor of shape (batch_size, 1).

    Returns:
      Output bounds.
    """
    return IntervalBounds(
        common.average_over_sequence(denom_for_avg, self.lower_rel),
        common.average_over_sequence(denom_for_avg, self.upper_rel),
        common.average_over_sequence(denom_for_avg, self.nominal))

  def apply_linear(self, wrapper, w, b):
    """Propagates the bounds through a linear layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 2D tensor of shape (input_size, output_size) containing
        weights for the linear layer.
      b: 1D tensor of shape (output_size) containing biases for the linear
        layer, or `None` if no bias.

    Returns:
      Output bounds.
    """
    lb, ub = linear_bounds(w, None, self.lower_rel, self.upper_rel)

    nominal_out = tf.matmul(self.nominal, w)
    if b is not None:
      nominal_out += b

    return IntervalBounds(lb, ub, nominal_out)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    """Propagates the bounds through a 1D convolution layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 3D tensor of shape (kernel_length, input_channels, output_channels)
        containing weights for the convolution.
      b: 1D tensor of shape (output_channels) containing biases for the
        convolution, or `None` if no bias.
      padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
      stride: Integer stride.

    Returns:
      Output bounds.
    """
    lb, ub = conv1d_bounds(w, None, padding, stride,
                           self.lower_rel, self.upper_rel)

    nominal_out = tf.nn.conv1d(self.nominal, w,
                               padding=padding, stride=stride)
    if b is not None:
      nominal_out += b

    return IntervalBounds(lb, ub, nominal_out)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    """Propagates the bounds through a 2D convolution layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 4D tensor of shape (kernel_height, kernel_width, input_channels,
        output_channels) containing weights for the convolution.
      b: 1D tensor of shape (output_channels) containing biases for the
        convolution, or `None` if no bias.
      padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
      strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

    Returns:
      Output bounds.
    """
    lb, ub = conv2d_bounds(w, None, padding, strides,
                           self.lower_rel, self.upper_rel)

    nominal_out = tf.nn.convolution(self.nominal, w,
                                    padding=padding, strides=strides)
    if b is not None:
      nominal_out += b

    return IntervalBounds(lb, ub, nominal_out)

  def apply_avgpool(self, module, kernel_shape, strides):
    return IntervalBounds(module(self.lower_rel), module(self.upper_rel),
                          module(self.nominal))

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    """Propagates the bounds through a non-linear activation layer or `add` op.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      fn: String specifying non-linear activation function.
        May be one of: sig, relu, tanh, elu, leaky_relu.
        Anything else denotes identity.
      *args: Other inputs' bounds, for a multi-input node (e.g. Add).
      **parameters: Optional parameters if activation is parameterised, e.g.
        `{'alpha': 0.2}` for leaky ReLu.

    Returns:
      Output bounds.
    """
    if fn.__name__ in ('add', 'reduce_mean', 'avg_pool'):
      return IntervalBounds(
          fn(self.lower_rel, *[bounds.lower_rel for bounds in args]),
          fn(self.upper_rel, *[bounds.upper_rel for bounds in args]),
          fn(self.nominal, *[bounds.nominal for bounds in args]))
    else:
      assert not args, 'unary function expected'
      nominal_out = fn(self.nominal)
      if fn.__name__ == 'reduce_max':
        lb, ub = maxpool_bounds(fn, None, None,
                                self.lower_rel, self.upper_rel,
                                nominal_in=self.nominal,
                                nominal_out=nominal_out)
      elif fn.__name__ == 'max_pool':
        lb, ub = maxpool_bounds(fn,
                                parameters['ksize'][1:-1],
                                parameters['strides'][1:-1],
                                self.lower_rel, self.upper_rel,
                                nominal_in=self.nominal,
                                nominal_out=nominal_out)
      else:
        lb, ub = activation_bounds(fn, self.lower_rel, self.upper_rel,
                                   nominal_in=self.nominal,
                                   parameters=parameters)
      return IntervalBounds(lb, ub, nominal_out)

  def apply_maxpool(self, module, kernel_shape, strides):
    nominal_out = module(self.nominal)
    lb, ub = maxpool_bounds(module, kernel_shape, strides,
                            self.lower_rel, self.upper_rel,
                            nominal_in=self.nominal, nominal_out=nominal_out)
    return IntervalBounds(lb, ub, nominal=nominal_out)

  def apply_batch_norm(self, wrapper, mean, variance, scale, bias, epsilon):
    """Propagates the bounds through a batch norm layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      mean: Learnt batch mean.
      variance: Learnt batch variance.
      scale: Trained component-wise scale variable.
      bias: Trained component-wise bias variable.
      epsilon: Epsilon for avoiding instability when `variance` is very small.

    Returns:
      Output bounds.
    """
    lb, ub = batchnorm_bounds(mean, variance, scale, bias, epsilon,
                              self.lower_rel, self.upper_rel, is_relative=True)
    nominal_out = tf.nn.batch_normalization(self.nominal,
                                            mean, variance,
                                            bias, scale, epsilon)
    return IntervalBounds(lb, ub, nominal_out)

  def _set_up_cache(self):
    self._lower_rel, update_lower = self._cache_with_update_op(self._lower_rel)
    self._upper_rel, update_upper = self._cache_with_update_op(self._upper_rel)
    return tf.group([update_lower, update_upper])


def linear_bounds(w, b, lb_in, ub_in):
  """Calculates naive bounds on output of a linear layer.

  Args:
    w: 2D tensor of shape (input_size, output_size) containing
      weights for the linear layer.
    b: 1D tensor of shape (output_size) containing biases for the linear
      layer, or `None` if no bias.
    lb_in: 2D tensor of shape (batch_size, input_size) containing lower bounds
      on the inputs to the linear layer.
    ub_in: 2D tensor of shape (batch_size, input_size) containing upper bounds
      on the inputs to the linear layer.

  Returns:
    lb_out: 2D tensor of shape (batch_size, output_size) with lower bounds
      on the outputs of the linear layer.
    ub_out: 2D tensor of shape (batch_size, output_size) with upper bounds
      on the outputs of the linear layer.
  """
  weight_pos = tf.maximum(w, 0)
  weight_neg = tf.minimum(w, 0)
  lb_out = tf.matmul(lb_in, weight_pos) + tf.matmul(ub_in, weight_neg)
  ub_out = tf.matmul(ub_in, weight_pos) + tf.matmul(lb_in, weight_neg)
  if b is not None:
    lb_out += b
    ub_out += b
  return lb_out, ub_out


def conv2d_bounds(w, b, padding, strides, lb_in, ub_in):
  """Calculates naive bounds on output of a 2D convolution layer.

  Args:
    w: 4D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    b: 1D tensor of shape (output_channels) containing biases for the
      convolution, or `None` if no bias.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    lb_in: 4D tensor of shape (batch_size, input_height, input_width,
      input_channels) containing lower bounds on the inputs to the
      convolution layer.
    ub_in: 4D tensor of shape (batch_size, input_height, input_width,
      input_channels) containing upper bounds on the inputs to the
      convolution layer.

  Returns:
    lb_out: 4D tensor of shape (batch_size, output_height, output_width,
      output_channels) with lower bounds on the outputs of the
      convolution layer.
    ub_out: 4D tensor of shape (batch_size, output_height, output_width,
      output_channels) with upper bounds on the outputs of the
      convolution layer.
  """
  weight_pos = tf.maximum(w, 0)
  weight_neg = tf.minimum(w, 0)
  lb_out = (tf.nn.convolution(lb_in, weight_pos,
                              padding=padding, strides=strides) +
            tf.nn.convolution(ub_in, weight_neg,
                              padding=padding, strides=strides))
  ub_out = (tf.nn.convolution(ub_in, weight_pos,
                              padding=padding, strides=strides) +
            tf.nn.convolution(lb_in, weight_neg,
                              padding=padding, strides=strides))
  if b is not None:
    lb_out += b
    ub_out += b
  return lb_out, ub_out


def conv1d_bounds(w, b, padding, stride, lb_in, ub_in):
  """Calculates naive bounds on output of a 1D convolution layer.

  Args:
    w: 3D tensor of shape (kernel_length, input_channels, output_channels)
      containing weights for the convolution.
    b: 1D tensor of shape (output_channels) containing biases for the
      convolution, or `None` if no bias.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    stride: Integer stride.
    lb_in: 3D tensor of shape (batch_size, input_length, input_channels)
      containing lower bounds on the inputs to the convolution layer.
    ub_in: 3D tensor of shape (batch_size, input_length, input_channels)
      containing upper bounds on the inputs to the convolution layer.

  Returns:
    lb_out: 3D tensor of shape (batch_size, output_length, output_channels)
      with lower bounds on the outputs of the convolution layer.
    ub_out: 3D tensor of shape (batch_size, output_length, output_channels)
      with upper bounds on the outputs of the convolution layer.
  """
  weight_pos = tf.maximum(w, 0)
  weight_neg = tf.minimum(w, 0)
  lb_out = (tf.nn.conv1d(lb_in, weight_pos, padding=padding, stride=stride) +
            tf.nn.conv1d(ub_in, weight_neg, padding=padding, stride=stride))
  ub_out = (tf.nn.conv1d(ub_in, weight_pos, padding=padding, stride=stride) +
            tf.nn.conv1d(lb_in, weight_neg, padding=padding, stride=stride))
  if b is not None:
    lb_out += b
    ub_out += b
  return lb_out, ub_out


def batchnorm_bounds(mean, variance, scale, bias, epsilon,
                     lb_in, ub_in, is_relative=False):
  """Calculates naive bounds on the output of a BatchNorm layer.

  The BatchNorm will be applied using its current moving averages but without
  updating them.

  Args:
    mean: Learnt batch mean.
    variance: Learnt batch variance.
    scale: Trained component-wise scale variable.
    bias: Trained component-wise bias variable.
    epsilon: Epsilon for avoiding instability when `variance` is very small.
    lb_in: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      containing lower bounds on the inputs to the batch norm layer.
    ub_in: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      containing upper bounds on the inputs to the batch norm layer.
    is_relative: Whether to ignore all bias terms, effectively expressing the
      returned output bounds relative to `nominal_out=batchnorm(nominal_in)`.
      Defaults to False.

  Returns:
    lb_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with lower bounds on the outputs of the batch norm layer.
    ub_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with upper bounds on the outputs of the batch norm layer.
  """
  lb_out = tf.nn.batch_normalization(
      lb_in,
      tf.zeros_like(mean) if is_relative else mean, variance,
      None if is_relative else bias, scale, epsilon)
  ub_out = tf.nn.batch_normalization(
      ub_in,
      tf.zeros_like(mean) if is_relative else mean, variance,
      None if is_relative else bias, scale, epsilon)

  # It's just possible that the batchnorm's scale is negative.
  lb_out, ub_out = tf.minimum(lb_out, ub_out), tf.maximum(lb_out, ub_out)
  return lb_out, ub_out


def maxpool_bounds(module, kernel_shape, strides, lb_in, ub_in,
                   nominal_in=None, nominal_out=None):
  """Calculates naive bounds on output of an N-D max pool layer.

  Args:
    module: Callable for max-pool operation.
    kernel_shape: Integer list of `[kernel_height, kernel_width]`,
      or `None` to aggregate over the layer`s entire spatial extent.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    lb_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing lower bounds on the inputs to the
      max pool layer.
    ub_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing upper bounds on the inputs to the
      max pool layer.
    nominal_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values.
      Inputs bounds are interpreted relative to this.
    nominal_out: (N+2)D tensor of shape (batch_size, output_height,output_width,
      layer_channels) containing nominal input values.
      The returned output bounds are expressed relative to this.

  Returns:
    lb_out: (N+2)D tensor of shape (batch_size, output_height, output_width,
      layer_channels) with lower bounds on the outputs of the max pool layer.
    ub_out: (N+2)D tensor of shape (batch_size, output_height, output_width,
      layer_channels) with upper bounds on the outputs of the max pool layer.
  """
  if kernel_shape is None:
    nominal_out = tf.reduce_max(nominal_in,
                                axis=list(range(1, nominal_in.shape.ndims-1)),
                                keepdims=True)
    return (module((nominal_in - nominal_out) + lb_in),
            module((nominal_in - nominal_out) + ub_in))
  else:
    # Must perform the max on absolute bounds, as the kernels may overlap.
    # TODO(stanforth) investigate a more numerically stable implementation
    del strides
    return (module(nominal_in + lb_in) - nominal_out,
            module(nominal_in + ub_in) - nominal_out)


def activation_bounds(nl_fun, lb_in, ub_in, nominal_in=None, parameters=None):
  """Calculates naive bounds on output of an activation layer.

  Inputs bounds are interpreted relative to `nominal_in`, and the returned
  output bounds are expressed relative to `nominal_out=nl(nominal_in)`.

  Args:
    nl_fun: Callable implementing the activation function itself.
    lb_in: (N+2)D tensor of shape (batch_size, layer_height, layer_width,
      layer_channels) containing lower bounds on the pre-activations.
    ub_in: (N+2)D tensor of shape (batch_size, layer_height, layer_width,
      layer_channels) containing upper bounds on the pre-activations.
    nominal_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values.
    parameters: Optional parameter dict if activation is parameterised, e.g.
      `{'alpha': 0.2}` for leaky ReLu.

  Returns:
    lb_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with lower bounds on the activations.
    ub_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with upper bounds on the activations.
  """
  if nl_fun.__name__ == 'relu':
    return (
        tf.maximum(tf.minimum(nominal_in, 0.) + lb_in,
                   tf.minimum(-nominal_in, 0.)),  # pylint:disable=invalid-unary-operand-type
        tf.maximum(tf.minimum(nominal_in, 0.) + ub_in,
                   tf.minimum(-nominal_in, 0.)))  # pylint:disable=invalid-unary-operand-type
  elif nl_fun.__name__ == 'leaky_relu':
    alpha = parameters['alpha']
    return (
        tf.maximum(
            lb_in + tf.minimum(nominal_in, 0.) * (1. - alpha),
            alpha * lb_in + tf.minimum(-nominal_in, 0.) * (1. - alpha)),  # pylint:disable=invalid-unary-operand-type
        tf.maximum(
            ub_in + tf.minimum(nominal_in, 0.) * (1. - alpha),
            alpha * ub_in + tf.minimum(-nominal_in, 0.) * (1. - alpha)))  # pylint:disable=invalid-unary-operand-type
  else:
    nominal_out = nl_fun(nominal_in)
    return (nl_fun(nominal_in + lb_in) - nominal_out,
            nl_fun(nominal_in + ub_in) - nominal_out)


def input_bounds(inputs, delta, lower_bound=0., upper_bound=1.,
                 preprocess_fn=None):
  """Calculates interval bounds on the network inputs.

  Args:
    inputs: 2D tensor of shape (batch_size, input_size), or 4D tensor of
      shape (batch_size, height, width, channels), of input examples.
    delta: Permitted perturbation on each input.
    lower_bound: Scalar - smallest permissible input (pixel) value.
    upper_bound: Scalar - largest permissible input (pixel) value.
    preprocess_fn: Optional function mapping tensor to tensor
      performing pre-processing on the raw inputs.

  Returns:
    `IntervalBounds` for the inputs, relative to `inputs`.
  """
  # Input range, according to permitted perturbation radius.
  if preprocess_fn:
    lb = preprocess_fn(tf.maximum(inputs - delta, lower_bound)) - inputs
    ub = preprocess_fn(tf.minimum(inputs + delta, upper_bound)) - inputs
  else:
    lb = tf.maximum(-delta, lower_bound - inputs)
    ub = tf.minimum(delta, upper_bound - inputs)
  return IntervalBounds(lb, ub, inputs)


class NaiveBoundPropagation(layer_bounds.BoundPropagation):
  """Naive layer-wise bound propagation method."""
