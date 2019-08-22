# coding=utf-8
# Copyright 2019 Deep-Verify Authors.
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

from deep_verify.src import common
import tensorflow as tf


def _reduce_softmax(v, axis, inverse_temperature=None):
  if inverse_temperature is None:
    return tf.reduce_max(v, axis=axis)
  # Apply smoothing.
  return tf.reduce_sum(v * tf.nn.softmax(inverse_temperature * v, axis=axis),
                       axis=axis)


def _conj(nl, mu, lam, nominal, lb, ub, parameters=None,
          inverse_temperature=None):
  """Dual objective contribution of a pre-activation value.

  Finds the maximum value of `mu*x - lam*h(x)` for `lb <= x <= ub`
  where `h` is the activation function.

  If `nominal` is not `None`, then inputs and activations are interpreted
  relative to nominal inputs and outputs respectively, so we actually maximise
  `mu*x - lam*(h(nominal+x) - h(nominal))`.

  Args:
    nl: Callable for the non-linear activation function, e.g. tf.nn.relu.
    mu: (N+2)D tensor of shape (num_classes, batch_size, *layer_shape)
      containing Lagrange multipliers for the neurons' linear calculations.
    lam: (N+2)D tensor of shape (num_classes, batch_size, *layer_shape)
      containing Lagrange multipliers for the neurons' non-linear activations.
    nominal: (N+1)D tensor of shape (batch_size, *layer_shape) containing
      nominal input values. Inputs bounds are interpreted relative to these
      nominal values. Defaults to zero if `None`.
    lb: (N+1)D tensor of shape (batch_size, *layer_shape) containing
      lower bounds of the neurons' pre-activation values.
    ub: (N+1)D tensor of shape (batch_size, *layer_shape) containing
      upper bounds of the neurons' pre-activation values.
    parameters: Optional parameter dict.
    inverse_temperature: Optional parameter to use a soft maximum. When
      set to zero maximum will become equivalent to an average (over the
      candidate points). When set to infinity (or None), it is equivalent to
      taking the maximum. Note that a temperature will not necessarily result
      in a valid verification bound. It approaches a valid bound as the
      inverse_temperature rises.

  Returns:
    (N+1)D tensor of shape (num_classes, batch_size, *layer_shape) containing
      maximum attained value of mu*x - lam*h(x).
  """
  # Endpoints of admissible x-range are candidates for the maximum.
  broadcast = tf.zeros_like(lam + mu)
  lam = lam + broadcast
  mu = mu + broadcast
  # Candidates (and bounds) are relative to nominal, if nominal is supplied.
  candidates = [broadcast + lb, broadcast + ub]
  if nominal is None:
    clamp = lambda x: tf.maximum(lb, tf.minimum(ub, x))
  else:
    # x is absolute (e.g. zero for ReLU).
    # Shift relative to nominal, before clamping to within relative bounds.
    clamp = lambda x: tf.maximum(lb, tf.minimum(ub, x - nominal))

  # Use calculus to determine candidates internal to the admissible x-range.
  if nl.__name__ == 'sigmoid':
    def siginv(y):
      cond_y = tf.logical_and(y > 0, y < 1)
      y = tf.minimum(1-1e-16, tf.maximum(1e-16, y))
      return tf.where(cond_y, tf.log(y) - tf.log(1 - y),
                      tf.ones_like(y) / 0.)
    ratio_cond = tf.abs(lam) > 1e-14
    ratio = lam / tf.where(ratio_cond, mu, tf.ones_like(mu))
    cond_lam = tf.logical_and(ratio_cond, ratio > 0)
    cond_lam = tf.logical_and(cond_lam, ratio < .25)
    sqrt = tf.sqrt(tf.where(cond_lam, 1 - 4 * ratio, tf.zeros_like(mu)))
    candidates.append(tf.where(
        cond_lam,
        clamp(siginv((1 - sqrt) / 2.0)),
        broadcast + lb))
    candidates.append(tf.where(
        cond_lam,
        clamp(siginv((1 + sqrt) / 2.0)),
        broadcast + ub))
  elif nl.__name__ == 'tanh':
    ratio_cond = tf.abs(mu) > 1e-14
    ratio = lam / tf.where(ratio_cond, mu, tf.ones_like(mu))
    cond_lam = tf.logical_and(ratio_cond, ratio > 1)
    sqrt = tf.sqrt(tf.maximum(ratio, 1)) + 1e-6
    candidates.append(tf.where(
        cond_lam,
        clamp(-tf.acosh(sqrt)),
        broadcast + lb))
    candidates.append(tf.where(
        cond_lam,
        clamp(tf.acosh(sqrt)),
        broadcast + ub))
  elif nl.__name__ == 'elu':
    ratio_cond = tf.abs(mu) > 1e-6
    ratio = lam / tf.where(ratio_cond, mu, tf.ones_like(mu))
    cond_lam = tf.logical_and(ratio_cond, ratio > 1)
    maximum = tf.maximum(ratio, 1.)
    candidates.append(tf.where(
        cond_lam,
        clamp(-tf.log(maximum)),
        broadcast + lb))
  elif nl.__name__ in ('relu', 'leaky_relu'):
    # Include zero in the candidates as potential max/min points.
    candidates.append(broadcast + clamp(tf.zeros_like(lb)))
  else:
    # For identity activation, consider the endpoints only.
    pass
  x = tf.stack(candidates, axis=0)

  if nominal is None:
    fun_vals = nl(x)
  elif nl == 'relu':
    # ReLU(a+x) - ReLU(a)  =  max(min(a, 0) + x, min(-a, 0))
    fun_vals = tf.maximum(tf.minimum(nominal, 0) + x,
                          tf.minimum(-nominal, 0))
  elif nl == 'leaky_relu':
    # LeakyReLU(a+x) - LeakyReLUReLU(a)  =
    # max(x + min(a, 0) * (1 - alpha), alpha * x + min(-a, 0) * (1 - alpha))
    alpha = parameters['alpha']
    fun_vals = tf.maximum(x + tf.minimum(nominal, 0.) * (1. - alpha),
                          alpha * x + tf.minimum(-nominal, 0.) * (1. - alpha))
  else:
    fun_vals = nl(nominal + x) - nl(nominal)

  v = mu * x - lam * fun_vals
  return _reduce_softmax(v, 0, inverse_temperature)


def max_linear(coefficients, lb, ub, axis=None, keepdims=False,
               inverse_temperature=None):
  """Maximises linear combinations over inputs within a specified hypercube.

  Args:
    coefficients: 3D tensor of shape (num_classes, batch_size, layer_size) or
      5D tensor of shape (num_classes, batch_size, input_height,
      input_width, input_channels)
      containing coefficients of the linear combinations.
    lb: 2D tensor of shape (batch_size, layer_size) or 4D tensor of shape
      (batch_size, input_height, input_width, input_channels) containing lower
      bounds on inputs.
    ub: 2D tensor of shape (batch_size, layer_size) or 4D tensor of shape
      (batch_size, input_height, input_width, input_channels) containing upper
      bounds on inputs.
    axis: Axis/axes (after broadcasting lb,ub) over which to take the linear
      combination, or `None` to default to 'all dims after the leading two'.
    keepdims: Whether to retain the dimensions over which the linear
      combination is taken.
    inverse_temperature: Optional parameter to use a soft maximum. When
      set to zero maximum will become equivalent to an average (over the
      candidate points). When set to infinity (or None), it is equivalent to
      taking the maximum. Note that a temperature will not necessarily result
      in a valid verification bound. It approaches a valid bound as the
      inverse_temperature rises.

  Returns:
    opt_val: 2D tensor of shape (num_classes, batch_size) containing the
      maximum attained values of the linear combinations.
  """
  if axis is None:
    axis = list(range(2, coefficients.shape.ndims))
  v = tf.stack([coefficients * lb, coefficients * ub], axis=0)
  v = _reduce_softmax(v, 0, inverse_temperature)
  return tf.reduce_sum(v, axis=axis, keepdims=keepdims)


def linear_dual_objective(lam_in, activation_coeffs, dual_obj_bias, lb, ub,
                          inverse_temperature=None):
  """Calculates contribution to dual objective for a linear/conv layer.

  Maximises (over x in [lb, ub])::
    lam_{l-1}^T x  -  mu_l^T w_l x  -  mu_l^T b_l

  Args:
    lam_in: 3D tensor of shape (num_classes, batch_size, input_size) or
      5D tensor of shape (num_classes, batch_size, input_height,
      input_width, input_channels)
      containing Lagrange multipliers for the input neurons' activations,
      or `None` if this is the first layer.
    activation_coeffs: 3D tensor of shape (num_classes, batch_size, layer_size)
      or 5D tensor of shape (num_classes, batch_size, input_height,
      input_width, input_channels) containing mu_l^T w_l.
    dual_obj_bias: 2D tensor of shape (num_classes, batch_size) containing
      mu_l^T b_l.
    lb: 2D tensor of shape (batch_size, layer_size) or 4D tensor of shape
      (batch_size, input_height, input_width, input_channels) containing lower
      bounds on inputs.
    ub: 2D tensor of shape (batch_size, layer_size) or 4D tensor of shape
      (batch_size, input_height, input_width, input_channels) containing upper
      bounds on inputs.
    inverse_temperature: Optional parameter to use a soft maximum. When
      set to zero maximum will become equivalent to an average (over the
      candidate points). When set to infinity (or None), it is equivalent to
      taking the maximum. Note that a temperature will not necessarily result
      in a valid verification bound. It approaches a valid bound as the
      inverse_temperature rises.

  Returns:
    2D tensor of shape (num_classes, batch_size) containing dual objective
      contribution for each example.
  """
  if lam_in is not None:
    activation_coeffs += lam_in

  return dual_obj_bias + max_linear(activation_coeffs, lb, ub,
                                    inverse_temperature=inverse_temperature)


def batchnorm_layer_dual_objective(batchnorm_module, mu_out):
  """Calculates the dual objective contribution for a batch norm layer.

  This contribution arises because the batch norm's shifts will effectively
  amend the previous layer's biases.

  Also returns a rescaled version of `mu_out`, which should be used as the
  input to the call to xxx_layer_dual_objective for the layer upon which the
  batch norm acts.

  Args:
    batchnorm_module: `snt.BatchNorm` module.
    mu_out: 3D tensor of shape (num_classes, batch_size, output_size)
      or 5D tensor of shape (num_classes, batch_size, output_height,
      output_width, output_channels) containing Lagrange multipliers
      for the output neurons.

  Returns:
    dual_obj: 2D tensor of shape (num_classes, batch_size) containing dual
      objective contribution from the batch norm for each example.
    mu_bn: Tensor of same shape as `mu_out` that has been scaled according to
      the batch norm.
  """
  w_bn, b_bn = common.decode_batchnorm(batchnorm_module)
  mu_bn = w_bn * mu_out
  dual_obj = -tf.reduce_sum(b_bn * mu_out,
                            axis=list(range(2, mu_out.shape.ndims)))
  return dual_obj, mu_bn


def _prod(lst):
  return functools.reduce(operator.mul, lst, 1)


def maxpool_layer_dual_objective(kernel_shape, strides, with_relu,
                                 mu_in, lam_out, lb, ub, nominal=None):
  """Calculates the contribution to the dual objective of an N-D max pool layer.

  Maximises (over y in [lb, ub])::
    mu_l^T y  -  lam_l^T h_l(y)
  where `h` is the specified max pool operation.

  If `nominal` is not `None`, then inputs and maxima are interpreted
  relative to nominal inputs and outputs respectively, so we actually maximise::
    mu_l^T y - lam_l^T (h_l(nominal+y) - h_l(nominal))`.

  This formulation only supports maxpools that cover the input space without
  gaps. Overlaps are permitted, although they will give rise to an overestimate
  of the dual objective rather than a tight value.

  Args:
    kernel_shape: Integer list of `[kernel_height, kernel_width]`,
      or `None` to aggregate over the layer`s entire spatial extent.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    with_relu: Whether to apply `tf.nn.relu` to the maxpool.
    mu_in: (N+3)D tensor of shape (num_classes, batch_size,
      input_height, input_width, layer_channels) containing
      Lagrange multipliers for the neurons' linear calculations.
    lam_out: (N+3)D tensor of shape (num_classes, batch_size,
      output_height, output_width, layer_channels) containing
      Lagrange multipliers for the neurons' maxpool calculations.
    lb: (N+2)D tensor of shape (batch_size,
      input_height, input_width, layer_channels) containing
      lower bounds of the neurons' pre-maxpool values.
    ub: (N+2)D tensor of shape (batch_size,
      input_height, input_width, layer_channels) containing
      upper bounds of the neurons' pre-maxpool values.
    nominal: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values. Inputs bounds are
      interpreted relative to these nominal values. Defaults to zero.

  Returns:
    2D tensor of shape (num_classes, batch_size) containing dual objective
      contribution for each example.

  Raises:
    ValueError: if the pools overlap or have gaps.
  """
  if nominal is not None:
    # TODO(stanforth) investigate a more numerically stable implementation
    res = maxpool_layer_dual_objective(kernel_shape, strides, with_relu,
                                       mu_in, lam_out,
                                       nominal + lb, nominal + ub)

    # Infer the nominal outputs.
    if kernel_shape is None:
      nominal_out = tf.reduce_max(nominal,
                                  axis=list(range(1, nominal.shape.ndims-1)))
    else:
      nominal_out = tf.nn.max_pool(nominal, ksize=kernel_shape, padding='VALID',
                                   strides=([1] + strides + [1]))
    if with_relu:
      nominal_out = tf.relu(nominal_out)

    res -= tf.reduce_sum(mu_in * nominal,
                         axis=list(range(2, mu_in.shape.ndims)))
    res += tf.reduce_sum(lam_out * nominal_out,
                         axis=list(range(2, lam_out.shape.ndims)))
    return res

  # Search for maximum by branching over inputs (kernel elements).

  # Broadcast the tensors to match what `fn` will operate with, i.e. shape
  # (num_classes, batch_size, output_height, output_width,
  #  kernel_height * kernel_width, layer_channels).

  num_classes = mu_in.shape[0].value
  batch_size = tf.shape(mu_in)[1]
  input_shape = mu_in.shape[2:].as_list()
  layer_channels = mu_in.shape[-1].value
  output_spatial_shape = lam_out.shape[2:-1].as_list()
  nd = lam_out.shape.ndims - 3

  if kernel_shape is None:
    # Maxpool will be across the entire layer (in each channel).
    kernel_size = _prod(input_shape[:-1])
    lb_bc = lb
    ub_bc = ub
    mu_bc = mu_in

  else:
    for i in range(len(kernel_shape)):
      if kernel_shape[i] < strides[i]:
        raise ValueError(
            'The pools must tile the entire input space without gaps.')
    padding = 'VALID'

    # Determine the fan-out of each input, where the pools overlap.
    # Builds a tensor of shape (1, 1, input_height, input_width, 1) of the form
    # [[1,1,2,1,1], [1,1,2,1,1], [2,2,4,2,2], [1,1,2,1,1], [1,1,2,1,1]]
    # (illustrated here with 3x3 kernel with stride 2 on a 5x5 input).
    overlap = common.conv_reduce_sum(
        tf.ones(dtype=mu_in.dtype, shape=(
            [1, 1] + output_spatial_shape + [1] + kernel_shape + [1])),
        input_shape,
        padding=padding, strides=strides)
    # Share mu values equally amongst pools where they overlap.
    mu_in /= overlap

    # Broadcast the bounds and mu vars where the kernel applications overlap.
    kernel_size = _prod(kernel_shape)
    lb_bc = common.conv_broadcast(lb, kernel_shape,
                                  padding=padding, strides=strides)
    ub_bc = common.conv_broadcast(ub, kernel_shape,
                                  padding=padding, strides=strides)
    # Temporarily combine the (num_classes, batch_size) dimensions
    # while applying the broadcast to mu.
    mu_bc = tf.reshape(mu_in, shape=([num_classes * batch_size] +
                                     mu_in.shape[2:].as_list()))
    mu_bc = common.conv_broadcast(mu_bc, kernel_shape,
                                  padding=padding, strides=strides)
    # conv_broadcast has returned tensors of shape
    # (N, output_height, output_width, 1, kernel_height, kernel_width, C).

  lb_bc = tf.reshape(lb_bc, shape=([1, batch_size] +
                                   output_spatial_shape +
                                   [kernel_size, layer_channels]))
  ub_bc = tf.reshape(ub_bc, shape=([1, batch_size] +
                                   output_spatial_shape +
                                   [kernel_size, layer_channels]))
  mu_bc = tf.reshape(mu_bc, shape=([num_classes, batch_size] +
                                   output_spatial_shape +
                                   [kernel_size, layer_channels]))
  lb_bc += tf.zeros_like(mu_bc)
  ub_bc += tf.zeros_like(mu_bc)

  # Use the same lambda for each input.
  lam_bc = tf.expand_dims(lam_out, axis=(nd+2))

  # All xx_bc tensors are shaped as (class, N, H, W, i, C)
  # where i ranges over inputs (kernel elements).

  # To calculate for input (kernel element) i, we need to sum over inputs j.
  # Set up xx_i, xx_j tensors shaped as (class, N, H, W, i, j, C)
  # where i,j both range over inputs (kernel elements).

  # y_i = tf.expand_dims(y, nd+3)  (will create inside `fn`)
  mu_j = tf.expand_dims(mu_bc, nd+2)
  lb_j = tf.expand_dims(lb_bc, nd+2)
  ub_j = tf.expand_dims(ub_bc, nd+2)
  # Only consider j != i.
  mask = 1.0 - tf.expand_dims(tf.eye(kernel_size), -1)

  def fn(y):
    """Optimal dual objective, conditional on the value of the maxpool.

    For each input (kernel element) i, for the given y_i,
    maximises (over z_j in [lb_j, min{y_i, ub_j}] and constraining z_i=y_i)::
      mu^T z  -  lam y_i

    This will be infeasible if y_i < lb_j for some j, (also if y_i < 0 in the
    case of relu+maxpool), so maxpool cannot be attained at input i. The
    returned tensor is unspecified for such elements.

    Args:
      y: (N+4)D tensor of shape (num_classes, batch_size,
        output_height, output_width,
        kernel_height * kernel_width, layer_channels) containing, for each
        input (kernel element) i, a value of maxpool assumed to be attained
        at input i.

    Returns:
      Tensor of same shape as `y` containing, for each input (kernel element) i,
        the optimal value of the dual objective, conditional the maxpool being
        equal to `y` with the maximum attained at input i.
    """
    y_i = tf.expand_dims(y, nd+3)
    # Maximise sum_{j!=i} mu_j y_j  where y_j <= y_i for all j!=i.
    obj = max_linear(mask * mu_j, lb_j, tf.minimum(ub_j, y_i), axis=(nd+3))
    return obj + (mu_bc - lam_bc) * y

  lb_max = tf.reduce_max(lb_bc, axis=(nd+2), keepdims=True)
  if with_relu:
    lb_max = tf.maximum(lb_max, 0.)
  _, attained = common.concave_max_binsearch(
      fn, tf.zeros_like(lb_bc) + lb_max, ub_bc)

  # Filter out any infeasible choices of i.
  attained = tf.where(lb_max <= ub_bc, attained, tf.zeros_like(attained) +
                      tf.reduce_min(attained, axis=(nd+2), keepdims=True))

  # Maximise over which input (kernel element) maximises the maxpool.
  per_neuron_objective = tf.reduce_max(attained, axis=(nd+2))

  if with_relu:
    # The relu+maxpool may additionally be 'maximised' by zero.
    # Calculate optimal dual objective, conditional on all y_i <= 0.
    # Maximise (over z_j in [lb_j, min{0, ub_j}])::
    #   mu^T z  -  lam 0
    attained_zero = max_linear(mu_bc, lb_bc, tf.minimum(ub_bc, 0.), axis=(nd+2))

    # Filter out any infeasible cases.
    per_neuron_objective = tf.where(
        tf.squeeze(lb_max, axis=(nd+2)) <= 0.,
        tf.maximum(per_neuron_objective, attained_zero),
        per_neuron_objective)

  return tf.reduce_sum(per_neuron_objective,
                       axis=list(range(2, per_neuron_objective.shape.ndims)))


def activation_layer_dual_objective(nl, mu_in, lam_out, lb, ub, nominal=None,
                                    parameters=None, inverse_temperature=None):
  """Calculates the contribution to the dual objective of an activation layer.

  Maximises (over y in [lb, ub])::
    mu_l^T y  -  lam_l^T h_l(y)
  where `h` is the specified non-linearity.

  If `nominal` is not `None`, then inputs and activations are interpreted
  relative to nominal inputs and outputs respectively, so we actually maximise::
    mu_l^T y - lam_l^T (h_l(nominal+y) - h_l(nominal))`.

  Args:
    nl: Callable for the non-linear activation function, e.g. tf.nn.relu.
    mu_in: (N+3)D tensor of shape (num_classes, batch_size,
      input_height, input_width, layer_channels) containing
      Lagrange multipliers for the neurons' linear calculations.
    lam_out: (N+3)D tensor of shape (num_classes, batch_size,
      output_height, output_width, layer_channels) containing
      Lagrange multipliers for the neurons' non-linear activations.
    lb: (N+2)D tensor of shape (batch_size,
      layer_height, layer_width, layer_channels)
      containing lower bounds of the neurons' pre-activation values.
    ub: (N+2)D tensor of shape (batch_size,
      layer_height, layer_width, layer_channels)
      containing upper bounds of the neurons' pre-activation values.
    nominal: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values. Inputs bounds are
      interpreted relative to these nominal values. Defaults to zero.
    parameters: Optional parameter dict.
    inverse_temperature: Optional parameter to use a soft maximum. When
      set to zero maximum will become equivalent to an average (over the
      candidate points). When set to infinity (or None), it is equivalent to
      taking the maximum. Note that a temperature will not necessarily result
      in a valid verification bound. It approaches a valid bound as the
      inverse_temperature rises.

  Returns:
    2D tensor of shape (num_classes, batch_size) containing dual objective
      contribution for each example.
  """
  per_neuron_objective = _conj(nl, mu_in, lam_out, nominal, lb, ub,
                               parameters=parameters,
                               inverse_temperature=inverse_temperature)
  return tf.reduce_sum(per_neuron_objective,
                       axis=list(range(2, per_neuron_objective.shape.ndims)))


