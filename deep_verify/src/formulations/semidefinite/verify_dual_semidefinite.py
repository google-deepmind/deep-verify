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

"""Graph construction for semidefinite formulation of dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.formulations import verify_dual_base
from deep_verify.src.formulations.semidefinite import gram_calcs
from deep_verify.src.layers import layers
from deep_verify.src.layers import layers_combined
from enum import Enum
import sonnet as snt
import tensorflow as tf


# Margin to ensure that Hessian remains positive definite in the face of
# numerical accuracy issues.
_EPSILON = 1.e-12


class VerifyOption(str, Enum):
  SVD = 'svd'
  WEAK = 'weak'
  STRONG = 'strong'
  STRONG_APPROX = 'strong_approx'


class _InputLayer(layers.VerifiableLayer):
  """No-op layer allowing dual variables at the first layer inputs."""

  def __init__(self, first_layer):
    if first_layer.is_activation:
      raise ValueError('First layer must not be an activation layer')
    super(_InputLayer, self).__init__()
    self._first_layer = first_layer

  @property
  def input_node(self):
    return self._first_layer.input_node

  @property
  def output_node(self):
    return self._first_layer.input_node

  @property
  def reshape(self):
    return self._first_layer.reshape

  def dual_shape(self):
    output_shape = super(_InputLayer, self).dual_shape()
    return {
        'x': output_shape,
    }

  def reshape_duals_forwards(self, next_layer, dual_vars):
    if next_layer.reshape:
      # There was a reshape prior to the next layer.
      reshape = snt.BatchReshape(next_layer.input_shape, preserve_dims=2)
      dual_vars = {key: reshape(dual_var)
                   for key, dual_var in dual_vars.items()}
    return dual_vars

  def project_duals_op(self, dual_vars):
    """Projects duals into their regional constraints.

    Args:
      dual_vars: Dual variable tensor.

    Returns:
      Tensor of same type and shape as `dual_vars`, in which the dual variable
      values are clamped to their admissible ranges (if relevant).
    """
    assign_ops = [tf.no_op()]

    return tf.group(*assign_ops)


class _CombinedLayer(layers.VerifiableLayer):
  """Linear layer followed by activation layer, treated together."""

  def __init__(self, linear_layer, activation_layer):
    if linear_layer.is_activation:
      raise ValueError('First layer must not be an activation layer')
    if not activation_layer.is_activation:
      raise ValueError('Second layer must be an activation layer')
    super(_CombinedLayer, self).__init__()
    self._linear_layer = linear_layer
    self._activation_layer = activation_layer

  @property
  def linear_layer(self):
    return self._linear_layer

  @property
  def activation_layer(self):
    return self._activation_layer

  @property
  def input_node(self):
    return self._linear_layer.input_node

  @property
  def output_node(self):
    return self._activation_layer.output_node

  @property
  def reshape(self):
    return self._linear_layer.reshape

  def dual_shape(self):
    output_shape = super(_CombinedLayer, self).dual_shape()
    input_shape = tuple(self.input_shape)
    return {
        'x': output_shape,
        'delta': output_shape,
        'beta': input_shape,
        'lambda': output_shape,
        'mup': output_shape,
        'mum': output_shape,
        'muc': output_shape
    }

  def reshape_duals_forwards(self, next_layer, dual_vars):
    if next_layer.reshape:
      # There was a reshape prior to the next layer.
      reshape = snt.BatchReshape(next_layer.input_shape, preserve_dims=2)
      dual_vars = {key: reshape(dual_var) if key != 'beta' else dual_var
                   for key, dual_var in dual_vars.items()}
    return dual_vars

  def project_duals_op(self, dual_vars):
    """Projects duals into their regional constraints.

    Args:
      dual_vars: Dual variable tensor.

    Returns:
      Tensor of same type and shape as `dual_vars`, in which the dual variable
      values are clamped to their admissible ranges (if relevant).
    """
    beta = dual_vars['beta']
    mup, mum, muc = dual_vars['mup'], dual_vars['mum'], dual_vars['muc']

    assign_ops = []
    # mup, mum, muc must be non-negative
    lb_pre = self._activation_layer.input_bounds.lower
    ub_pre = self._activation_layer.input_bounds.upper
    mu_shape = mup.get_shape().as_list()
    tile_shape = [mu_shape[0]] + [1 for _ in mu_shape[1:]]
    on = tf.tile(
        tf.expand_dims(lb_pre > 0.0, 0),
        tile_shape)
    off = tf.tile(
        tf.expand_dims(ub_pre < 0.0, 0),
        tile_shape)
    known = tf.logical_or(on, off)

    assign_ops.append(tf.assign(mup,
                                tf.where(on, mup,
                                         tf.where(off, tf.zeros_like(mup),
                                                  tf.maximum(mup, 0.)))))
    assign_ops.append(tf.assign(
        mum, tf.where(known, tf.zeros_like(mum), tf.maximum(mum, 0.))))
    assign_ops.append(tf.assign(
        muc, tf.where(known, tf.zeros_like(muc), tf.maximum(muc, 0.))))
    # beta must be strictly positive.
    assign_ops.append(tf.assign(beta, tf.maximum(beta, 1.)))

    return tf.group(*assign_ops)


class SemidefiniteDualFormulation(verify_dual_base.DualFormulation):
  """Specifies layers' dual verification contributions."""

  def group_layers(self, verifiable_layers):
    """Groups dual layers as required by the verification strategy.

    Args:
      verifiable_layers: List of `SingleVerifiableLayer` objects specifying
        linear layers and non-linear activation functions.

    Returns:
      List of `VerifiableLayer` objects specifying layers that give rise to
      dual variables.

    Raises:
      ValueError: if an unsupported layer type or arrangement is encountered.
    """
    verifiable_layers = layers_combined.combine_trailing_linear_layers(
        verifiable_layers)
    if len(verifiable_layers) < 3:
      raise ValueError('The last predictor layer must be a linear layer. The '
                       'predictor must also contain at least one '
                       'linear-like layer followed by a non-linearity.')

    # Group the layers.
    grouped_layers = []

    # Start with a no-op shaped according to the first linear layer's inputs.
    grouped_layers.append(_InputLayer(verifiable_layers[0]))

    # Layers are grouped in (activation, linear) pairs.
    l = 0
    while l < len(verifiable_layers) - 1:
      grouped_layers.append(_CombinedLayer(
          verifiable_layers[l],
          verifiable_layers[l+1]))
      l += 2

    # Final layer (linear). It has no duals.
    if l >= len(verifiable_layers):
      raise ValueError('This formulation requires the network to end '
                       'linear -> activation -> linear.')
    grouped_layers.append(verifiable_layers[l])
    grouped_layers[-1].set_no_duals()
    return grouped_layers

  def set_objective_computation_config(self, config):
    if config is None:
      config = {}
    self._softplus_temperature = config.get('softplus_temperature', 1.)
    self._verify_option = config.get('verify_option', VerifyOption.WEAK)
    self._sqrt_eps = config.get('sqrt_eps', _EPSILON)
    self._approx_k = config.get('approx_k', 100)
    self._exact_k = config.get('exact_k', 1000)
    self._soften_abs = config.get('soften_abs', False)
    self._use_lam = config.get('use_lam', True)

  def _get_dual_vars(self, dual_vars):
    if dual_vars.get('lambda') is not None:
      lam = dual_vars.get('lambda')
      delta = dual_vars['delta']
      alpha = tf.sqrt(tf.square(lam) + tf.square(delta) + self._sqrt_eps)
      return ((alpha + delta)/2, (alpha-delta)/2, lam/2)
    else:
      return (None, tf.zeros_like(dual_vars['x']), None)

  def expand_tensor(self, x, aug_size):
    return tf.reshape(x, aug_size + [-1])

  def dual_objective(self, verifiable_layers, labels, dual_var_lists,
                     target_strategy=None, objective_computation_config=None):
    """Computes the Lagrangian (dual objective).

    Args:
      verifiable_layers: List of `VerifiableLayer` objects specifying layers
        that give rise to dual variables.
      labels: 1D integer tensor of shape (batch_size) of labels for each
        input example.
      dual_var_lists: Nested list of 3D tensors of shape
        (num_classes, batch_size, layer_size) and 5D tensors of shape
        (num_classes, batch_size, height, width, channels)
        containing Lagrange multipliers for the layers' calculations.
        This has the same length as `verifiable_layers`, and typically each
        entry is a singleton list with the dual variable for that layer.
        ResNet blocks' entries have instead the structure
        [[left-sub-duals], [right-sub-duals], overall-dual].
      target_strategy: a `TargetObjective` object which gets the objective
        weights of the final layer depending on the strategy of the final
        objective. Default is set to None in which case it uses the default
        target strategy.
      objective_computation_config: `ConfigDict` of additional parameters.

    Returns:
      2D tensor of shape (num_classes, batch_size) containing dual objective
        for each target class, for each example.
    """
    self.set_objective_computation_config(objective_computation_config)
    batch_size = tf.shape(labels)[0]
    aug_batch_size = dual_var_lists[0][-1]['x'].shape.as_list()[:2]
    dtype = verifiable_layers[-1].output_bounds.lower.dtype

    # create dummy variables, only used to get gradient of Lagrangian wrt x
    dxs = []
    for layer in verifiable_layers[:-1]:
      dxs.append(tf.zeros(aug_batch_size + layer.output_shape))
    objective = tf.zeros([1, batch_size], dtype=dtype)
    # Contribution for the layers.
    lb_x = []
    ub_x = []
    for l in range(1, len(verifiable_layers) - 1):
      dual_vars_lm1 = verifiable_layers[l-1].reshape_duals_forwards(
          verifiable_layers[l], dual_var_lists[l-1][-1])
      dual_vars_l = dual_var_lists[l][-1]
      obj, lb_l, ub_l = self._layer_contrib(
          verifiable_layers[l], dual_vars_lm1, dual_vars_l,
          dxs[l-1],
          dxs[l])
      lb_x += [lb_l]
      ub_x += [ub_l]
      objective += obj

    last_layer = verifiable_layers[-1]  # linear layer.
    objective_weights = last_layer.get_objective_weights(
        labels, target_strategy=target_strategy)

    # Contribution for the final linear layer.
    dual_vars_km1 = verifiable_layers[-2].reshape_duals_forwards(
        last_layer, dual_var_lists[-2][-1])
    obj, lb_fin, ub_fin = self._last_layer_contrib(
        last_layer, dual_vars_km1,
        dxs[-1],
        objective_weights.w, objective_weights.b)
    lb_x += [lb_fin]
    ub_x += [ub_fin]
    objective += obj
    grad = tf.gradients(tf.reduce_sum(objective), dxs)
    for l, g in enumerate(grad):
      objective += tf.reduce_sum(
          g * tf.reshape((lb_x[l] + ub_x[l])/2, g.shape) +
          self._soft_abs(g * tf.reshape((ub_x[l] - lb_x[l])/2, g.shape)),
          axis=list(range(2, g.shape.ndims)))
    return objective

  def _layer_contrib(self, layer, dual_vars_lm1, dual_vars_l,
                     dx_lm1, dx_l):
    assert isinstance(layer, _CombinedLayer)

    if layer.activation_layer.activation != 'relu':
      raise NotImplementedError('Only ReLU nonlinearities are supported.')

    if layer.linear_layer.batch_norm is not None:
      raise NotImplementedError('BatchNorm is not yet implemented.')

    if self._use_lam:
      x_lm1 = dual_vars_lm1['x']
      x_l = dual_vars_l['x']
    else:
      x_lm1 = tf.zeros_like(dual_vars_lm1['x'])
      x_l = tf.zeros_like(dual_vars_l['x'])

    x_lm1 = x_lm1 + tf.reshape(dx_lm1, x_lm1.shape)

    x_l = x_l + tf.reshape(dx_l, x_l.shape)

    if self._use_lam:
      _, delta_lm1, lam_lm1 = self._get_dual_vars(dual_vars_lm1)

      alpha_l, _, lam_l = self._get_dual_vars(dual_vars_l)
      beta_l = dual_vars_l['beta']
    mup_l, mum_l, muc_l = (
        dual_vars_l['mup'], dual_vars_l['mum'], dual_vars_l['muc'])

    lb_lm1 = layer.input_bounds.lower
    ub_lm1 = layer.input_bounds.upper
    rad_lm1 = (ub_lm1 - lb_lm1) / 2.
    mid_lm1 = (ub_lm1 + lb_lm1) / 2.

    lb_pre = layer.activation_layer.input_bounds.lower
    ub_pre = layer.activation_layer.input_bounds.upper
    # If upper bound and lower bound are the same, d_l can be zero.
    same_pre = ub_pre - lb_pre < 1.e-8
    d_l = (
        tf.where(same_pre, tf.zeros_like(ub_pre), ub_pre) /
        tf.where(same_pre, tf.ones_like(lb_pre), ub_pre - lb_pre))
    d_l = tf.where(lb_pre > 0.0, tf.ones_like(lb_pre), d_l)
    d_l = tf.where(ub_pre < 0.0, tf.zeros_like(lb_pre), d_l)
    bias_l = (tf.where(same_pre, tf.zeros_like(ub_pre), ub_pre * lb_pre) /
              tf.where(same_pre, tf.ones_like(lb_pre), ub_pre - lb_pre))
    bias_l = tf.where(lb_pre > 0.0, tf.zeros_like(lb_pre), bias_l)
    bias_l = tf.where(ub_pre < 0.0, tf.zeros_like(ub_pre), bias_l)

    # Pre-activations.
    y_l = layer.linear_layer.forward_prop(x_lm1, apply_bias=True)

    # Layer-specific computation of norm(w).
    if self._use_lam:
      w_norm = self._calc_w_norm(layer.linear_layer, x_lm1, beta_l, alpha_l)

      # Contribution for nu.
      nu_lm1 = (
          (delta_lm1 if lam_lm1 is None else delta_lm1 - lam_lm1) +
          w_norm / 4.)
      nu_lm1 = nu_lm1/2 + self._soft_abs(nu_lm1)/2
      dual_obj = tf.reduce_sum(nu_lm1 * (rad_lm1 * rad_lm1 -
                                         (x_lm1-mid_lm1) * (x_lm1-mid_lm1)),
                               axis=list(range(2, nu_lm1.shape.ndims)))
      # Contribution for lambda.
      dual_obj += tf.reduce_sum(lam_l * x_l * (y_l - x_l),
                                axis=list(range(2, lam_l.shape.ndims)))
    else:
      dual_obj = tf.zeros(mup_l.shape.as_list()[:2])

    # Contribution for mu.
    dual_obj += tf.reduce_sum(mup_l * (x_l - y_l),
                              axis=list(range(2, mup_l.shape.ndims)))
    dual_obj += tf.reduce_sum(mum_l * x_l,
                              axis=list(range(2, mum_l.shape.ndims)))
    dual_obj += tf.reduce_sum(muc_l * (d_l * y_l - bias_l - x_l),
                              axis=list(range(2, muc_l.shape.ndims)))

    return dual_obj, lb_lm1-x_lm1, ub_lm1-x_lm1

  def _last_layer_contrib(self, layer, dual_vars_lm1, dx_lm1, obj_w, obj_b):

    if self._use_lam:
      x_lm1 = dual_vars_lm1['x']
    else:
      x_lm1 = tf.zeros_like(dual_vars_lm1['x'])
    x_lm1 = x_lm1 + tf.reshape(dx_lm1, x_lm1.shape)
    lb_lm1 = layer.input_bounds.lower
    ub_lm1 = layer.input_bounds.upper

    if self._use_lam:
      _, delta_lm1, lam_lm1 = self._get_dual_vars(dual_vars_lm1)

      rad_lm1 = (ub_lm1 - lb_lm1) / 2.
      mid_lm1 = (ub_lm1 + lb_lm1) / 2.

      # Contribution for nu.
      nu_lm1 = tf.nn.relu(
          delta_lm1 if lam_lm1 is None else delta_lm1 - lam_lm1)
      dual_obj = tf.reduce_sum(nu_lm1 * (rad_lm1 * rad_lm1 -
                                         (x_lm1-mid_lm1) * (x_lm1-mid_lm1)),
                               axis=list(range(2, nu_lm1.shape.ndims)))
    else:
      dual_obj = tf.zeros(x_lm1.shape.as_list()[:2])

    # Contribution for primal objective.
    dual_obj += tf.reduce_sum(obj_w * x_lm1, axis=2)
    dual_obj += obj_b

    return dual_obj, lb_lm1-x_lm1, ub_lm1-x_lm1

  def _calc_w_norm(self, layer, x_lm1, beta_l, alpha_l):
    # Norm of w, used for computing nu_lm1.
    if self._verify_option == VerifyOption.WEAK:
      w_norm = layer.forward_prop(beta_l, w_fn=tf.abs) * (alpha_l)
      w_norm = layer.backward_prop(w_norm, w_fn=tf.abs) / beta_l
    elif self._verify_option == VerifyOption.STRONG:
      w_norm = layer.custom_op(_ScaledGramComputation(self._soft_abs,
                                                      self._exact_k),
                               alpha_l, beta_l)
    else:
      # Linearise the convolution.
      flatten = snt.BatchFlatten(preserve_dims=2)
      unflatten = snt.BatchReshape(x_lm1.shape[2:].as_list(), preserve_dims=2)
      # flatten(beta_l): KxNxI   w_lin: IxO   flatten(delta_l,gam_l): KxNxO
      if self._verify_option == VerifyOption.SVD:
        w_lin, _ = layer.flatten()
        w_scaled = (tf.expand_dims(flatten(beta_l), -1) *
                    tf.expand_dims(tf.expand_dims(w_lin, 0), 0) *
                    tf.expand_dims(tf.sqrt(flatten(alpha_l) +
                                           self._sqrt_eps), 2))
        s = tf.svd(w_scaled, compute_uv=False)
        w_norm = tf.expand_dims(tf.reduce_max(s, axis=-1), axis=-1)
        w_norm = unflatten(w_norm*w_norm) / (beta_l * beta_l)
      elif self._verify_option == VerifyOption.STRONG_APPROX:
        # Get size of input to layer
        size_list = beta_l[0, 0, ...].shape.as_list()
        size_x = 1
        for s in size_list:
          size_x *= s

        # Prepare data
        shape_beta = beta_l.shape.as_list()[2:]
        batch_shape = beta_l.shape.as_list()[:2]
        shape_alpha = alpha_l.shape.as_list()[2:]
        beta_reduce = flatten(beta_l)
        beta_reduce = beta_reduce / tf.reduce_sum(beta_reduce, axis=2,
                                                  keepdims=True)
        beta_l = tf.reshape(beta_reduce, beta_l.shape)

        k_sample = min(self._approx_k, size_x)

        def process_columns(x, beta_cur):
          """Compute |W^T[alpha]Wx|beta_cur."""
          shape_x_batch = x.shape.as_list()[:3]
          x = tf.reshape(x, [shape_x_batch[0], -1] + shape_beta)
          x_prop = tf.reshape(layer.forward_prop(x),
                              shape_x_batch + shape_alpha)
          x = layer.backward_prop(
              tf.reshape(x_prop * tf.expand_dims(alpha_l, 2),
                         [shape_x_batch[0], -1] + shape_alpha)
              )
          x = tf.reshape(x, shape_x_batch + shape_beta)

          # Flatten beta and pick out relevant entries
          beta_reshape = beta_cur
          for _ in range(len(shape_beta)):
            beta_reshape = tf.expand_dims(beta_reshape, -1)
          return tf.reduce_sum(self._soft_abs(x) * beta_reshape, axis=2)

        # Accumulator for sum over columns
        samples = tf.random.categorical(tf.log(tf.reshape(beta_reduce,
                                                          [-1, size_x])+
                                               1e-10),
                                        k_sample)
        samples = tf.one_hot(tf.reshape(samples, [-1]), size_x, axis=-1)
        samples = tf.reshape(samples, (batch_shape + [k_sample] +
                                       shape_beta))
        x_acc = process_columns(samples,
                                tf.ones(batch_shape + [k_sample])/k_sample)
        w_norm = x_acc/beta_l
      else:
        raise ValueError('Unknown verification option: ' + self._verify_option)

    return w_norm

  def _soft_abs(self, x):
    if self._soften_abs:
      return tf.sqrt(tf.square(x) + self._sqrt_eps)
    else:
      return tf.abs(x)

  def _softplus(self, x):
    temperature = self._softplus_temperature
    return temperature * tf.nn.softplus(x / temperature)


class _ScaledGramComputation(layers.CustomOp):
  """Layer-specific op to compute weighted Gram matrix projections."""

  def __init__(self, abs_fn, block_size):
    super(_ScaledGramComputation, self).__init__()
    self._abs_fn = abs_fn
    self._block_size = block_size

  def visit_linear(self, layer, w, b, alpha, beta):
    return gram_calcs.linear_weighted_gram_abs_projection(
        w, alpha, beta,
        abs_fn=self._abs_fn,
        block_size=self._block_size)

  def visit_conv(self, layer, w, b, padding, strides, alpha, beta):
    # Specify a block size (in channels) of 1.
    # For convolutions, this was found to be the most efficient.
    return gram_calcs.conv_weighted_gram_abs_projection(
        w, alpha, beta,
        padding=padding, strides=strides,
        abs_fn=self._abs_fn,
        block_size=1)

  def visit_activation(self, layer):
    raise NotImplementedError()

