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

from deep_verify.src.formulations import verify_dual_base
from deep_verify.src.formulations.standard import standard_layer_calcs
from deep_verify.src.layers import layers
from deep_verify.src.layers import layers_combined
import tensorflow as tf


class StandardDualFormulation(verify_dual_base.DualFormulation,
                              layers.CustomOp):
  """Simplified standard/reduced formulation, with no ResNet block support."""

  def __init__(self, use_reduced=True):
    super(StandardDualFormulation, self).__init__()
    self._use_reduced = use_reduced

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
    # Locate the last activation layer.
    # It is usually at position len-2, but it's possible to be followed by
    # more than one linear layer.
    k = len(verifiable_layers) - 1
    while k >= 0 and not verifiable_layers[k].is_activation:
      k -= 1
    if k <= 0 or k >= len(verifiable_layers) - 1:
      raise ValueError('The last predictor layer must be a linear layer. The '
                       'predictor must also contain at least one '
                       'linear-like layer followed by a non-linearity.')

    # Group the layers.
    grouped_layers = []
    # Initial linear layer.
    initial_layer, l = self._get_single_layer(verifiable_layers, 0)
    grouped_layers.append(initial_layer)
    # Successive layers.
    self._group_next_layers(grouped_layers, verifiable_layers[l:k])

    # No duals for final layer (activation+linear).
    final_layer = verifiable_layers[k]
    for linear_layer in verifiable_layers[k+1:]:
      final_layer = layers_combined.CombinedLayer(final_layer, linear_layer)
    final_layer.set_no_duals()
    grouped_layers.append(final_layer)

    return grouped_layers

  def _group_next_layers(self, grouped_layers, verifiable_layers):
    """Groups dual layers as required by the verification strategy.

    Args:
      grouped_layers: Populated on exit with list of `VerifiableLayer` objects
        specifying layers that give rise to dual variables.
      verifiable_layers: List of `SingleVerifiableLayer` objects specifying
        linear layers and non-linear activation functions.

    Raises:
      ValueError: if an unsupported layer type or arrangement is encountered.
    """
    l = 0
    while l < len(verifiable_layers):
      l = self._group_next_layer(grouped_layers, verifiable_layers, l)

  def _group_next_layer(self, grouped_layers, verifiable_layers, l):
    """One step of dual layer grouping as required by the verification strategy.

    Args:
      grouped_layers: On exit, a new `VerifiableLayer` object will be appended,
        specifying a grouped layer that give rise to dual variables.
      verifiable_layers: List of `SingleVerifiableLayer` objects specifying
        linear layers and non-linear activation functions.
      l: Index within `verifiable_layers` at which to start grouping.

    Returns:
      Updated value of `l`: index within `verifiable_layers` up to which
        grouping has completed.
    """
    if self._use_reduced:
      # Activation layer followed by linear layer.
      activation_layer = verifiable_layers[l]
      l += 1
      if l == len(verifiable_layers):
        raise ValueError('This formulation requires the network to end '
                         'linear -> activation -> linear.')
      linear_layer, l = self._get_single_layer(verifiable_layers, l)
      grouped_layers.append(layers_combined.CombinedLayer(
          activation_layer,
          linear_layer))
      return l

    else:
      # Standard formulation. Linear and activation layers are kept single.
      layer, l = self._get_single_layer(verifiable_layers, l)
      grouped_layers.append(layer)
      return l

  def _get_single_layer(self, verifiable_layers, l):
    """Extracts a single layer, grouping consecutive linear layers.

    Args:
      verifiable_layers: List of `SingleVerifiableLayer` objects specifying
        linear layers and non-linear activation functions.
      l: Index within `verifiable_layers` at which to extract.

    Returns:
      layer: `layer`, with any grouping operations applied.
      l: Updated value of `l`: index within `verifiable_layers` up to which
        layers have been consumed.
    """
    layer = verifiable_layers[l]
    l += 1
    if isinstance(layer, layers.AffineLayer):
      while l < len(verifiable_layers) and isinstance(verifiable_layers[l],
                                                      layers.AffineLayer):
        layer = layers_combined.CombinedLinearLayer(layer, verifiable_layers[l])
        l += 1
    return self._single_layer(layer), l

  def _single_layer(self, layer):
    """Groups a single layer.

    Args:
      layer: layer to group.

    Returns:
      `layer`, with any grouping operations applied.
    """
    # This implementation is a no-op, but sub-classes will override this
    # to handle ResNet blocks (recursively grouping their child layers).
    return layer

  def _is_affine_next(self, layer):
    return (isinstance(layer, layers.AffineLayer) or
            isinstance(layer, layers_combined.CombinedLinearLayer))

  def set_objective_computation_config(self, config):
    self._inverse_temperature = None
    if config is not None:
      self._inverse_temperature = config.get(
          'inverse_temperature', float('inf'))
      if (isinstance(self._inverse_temperature, float) and
          self._inverse_temperature == float('inf')):
        self._inverse_temperature = None

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
    dtype = verifiable_layers[-1].output_bounds.lower.dtype

    # Use broadcasting. The last layer will force the objective shape to be
    # `[num_classes, batch_size]`.
    objective = tf.zeros([1, batch_size], dtype=dtype)

    # The Lagrangian is  L = sum_l max_x [lam_{l-1}^T x - lam_l^T h_l(x)]
    # where h_l(x) is layer l applied to values x, lam_{-1}=0.

    # For numerical stability, we actually calculate this as follows:
    # rearranged (but algebraically equivalent) way:
    #   L = sum_l max_x [lam_{l-1}^T (x - a_l) - lam_l^T (h_l(x) - h_l(a_l))]
    #     - lam_{final-1} a_{final}
    # where a_l are the nominal input values to layer l.
    # This rearranged form is equivalent because a_{l+1} = h_l(a_l).
    # Note: c = -lam_{final-1} are the objective weights.

    for l in range(len(verifiable_layers) - 1):
      if l == 0:
        dual_vars_lm1 = None
      else:
        dual_vars_lm1 = verifiable_layers[l-1].reshape_duals_forwards(
            verifiable_layers[l], dual_var_lists[l-1][-1])
      objective += self.layer_contrib(verifiable_layers[l],
                                      dual_vars_lm1, *dual_var_lists[l])

    # The negated objective weights take the role of dual variable for the
    # last layer.
    last_layer = verifiable_layers[-1]  # activation+linear combined.
    objective_weights = last_layer.linear_layer.get_objective_weights(
        labels, target_strategy=target_strategy)
    last_dual_var = last_layer.reshape_duals_backwards(-objective_weights.w)

    # Contribution for the final activation layer.
    dual_vars_lm1 = verifiable_layers[-2].reshape_duals_forwards(
        last_layer, dual_var_lists[len(verifiable_layers) - 2][-1])
    objective += self.layer_contrib(last_layer.activation_layer,
                                    dual_vars_lm1, last_dual_var)

    # Constant term (objective layer bias).
    objective += tf.reduce_sum(
        objective_weights.w * last_layer.linear_layer.input_bounds.nominal,
        axis=list(range(2, objective_weights.w.shape.ndims)))
    objective += objective_weights.b

    return objective

  def layer_contrib(self, layer, dual_vars_lm1, *dual_vars):
    """Computes the contribution of a layer to the dual objective."""
    if isinstance(layer, layers_combined.CombinedLayer):
      # Activation+Affine combination for the 'reduced' formulation.
      # Back-prop the duals through the affine layer.
      act_coeffs, obj_linear = self.affine_layer_act_coeffs(layer.linear_layer,
                                                            *dual_vars)
      lam_out = layer.reshape_duals_backwards(-act_coeffs)

      # Contribution for the activation layer.
      obj_activation = self.layer_contrib(layer.activation_layer,
                                          dual_vars_lm1, lam_out)
      return obj_linear + obj_activation

    elif self._is_affine_next(layer):
      activation_coeffs, dual_obj_bias = self.affine_layer_act_coeffs(
          layer, *dual_vars)

      # Compute:
      #   max_x (lam_{l-1}^T x  -  mu_l^T (W_l x + b_l))
      # where mu_l^T W_l is activation_coeffs
      # and mu_l^T b_l is dual_obj_bias.
      return self.affine_layer_contrib(layer, dual_vars_lm1,
                                       activation_coeffs, dual_obj_bias)

    else:
      # Compute the term
      #   max_y (mu_l^T y  -  lam_l^T h_l(y))
      # where h_l is the non-linearity for layer l.
      return self.nonlinear_layer_contrib(layer, dual_vars_lm1, *dual_vars)

  def affine_layer_act_coeffs(self, layer, *dual_vars):
    """Computes the coefficients W_l^T mu_l   and   b_l^T mu_l.

    These will later be used in the expression::
      max_x (lam_{l-1}^T x  -  mu_l^T (W_l x + b_l))
    where W_l, b_l is the affine mapping for layer l.

    Args:
      layer: affine layer, or ResNet block beginning/ending with affine layers.
      *dual_vars: mu_l, preceded by branch dual vars if it's a ResNet block.

    Returns:
      activation_coeffs: W_l^T mu_l
      dual_obj: b_l^T mu_l, the contribution to the dual objective
    """
    mu_l, = dual_vars
    mu_l = layer.backward_prop_batchnorm(mu_l)
    activation_coeffs = -layer.backward_prop(mu_l)
    # Objective contribution is zero, as we work relative to nominals.
    dual_obj = tf.zeros(tf.shape(mu_l)[:2], dtype=mu_l.dtype)
    return activation_coeffs, dual_obj

  def affine_layer_contrib(self, layer, dual_vars_lm1,
                           activation_coeffs, dual_obj_bias):
    """Computes the contribution of an affine layer to the dual objective.

    Compute the term::
      max_x (lam_{l-1}^T x  -  mu_l^T (W_l x + b_l))
    where W_l, b_l is the affine mapping for layer l.

    Args:
      layer: affine (linear/conv) layer.
      dual_vars_lm1: lam_{l-1}, or None for the first layer.
      activation_coeffs: mu_l^T W_l
      dual_obj_bias: mu_l^T b_l

    Returns:
      Dual objective contribution.
    """
    return standard_layer_calcs.linear_dual_objective(
        dual_vars_lm1,
        activation_coeffs, dual_obj_bias,
        layer.input_bounds.lower_rel, layer.input_bounds.upper_rel,
        inverse_temperature=self._inverse_temperature)

  def nonlinear_layer_contrib(self, layer, dual_vars_lm1, *dual_vars):
    """Computes the contribution of a non-linear layer to the dual objective.

    Compute the term
      max_y (mu_l^T y  -  lam_l^T h_l(y))
    where h_l is the non-linearity for layer l.

    Args:
      layer: non-linear layer, or ResNet block beginning with a non-linear
        layer.
      dual_vars_lm1: mu_{l-1}
      *dual_vars: lam_l, preceded by branch dual vars if it's a ResNet block.

    Returns:
      Dual objective contribution.
    """
    # Invoke visit_activation, visit_maxpool, or visit_resnet_block.
    return layer.custom_op(self, dual_vars_lm1, *dual_vars)

  def visit_activation(self, layer, mu_lm1, lam_l):
    return standard_layer_calcs.activation_layer_dual_objective(
        layer.module, mu_lm1, lam_l,
        layer.input_bounds.lower_rel, layer.input_bounds.upper_rel,
        nominal=layer.input_bounds.nominal, parameters=layer.parameters,
        inverse_temperature=self._inverse_temperature)

  def visit_maxpool(self, layer, mu_lm1, lam_l):
    self._ensure_no_temperature()
    return standard_layer_calcs.maxpool_layer_dual_objective(
        layer.module, mu_lm1, lam_l,
        layer.input_bounds.lower_rel, layer.input_bounds.upper_rel,
        nominal=layer.input_bounds.nominal)

  def _ensure_no_temperature(self):
    if self._inverse_temperature is not None:
      raise ValueError('Smoothing of the dual objective is not supported.')

