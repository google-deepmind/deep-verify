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

"""Defines composite layers with customised dual variables."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.layers import layers
import sonnet as snt
import tensorflow as tf


class CombinedLayer(layers.VerifiableLayer):
  """Activation plus linear layer, treated together (used by reduced)."""

  def __init__(self, activation_layer, linear_layer):
    if linear_layer.is_activation:
      raise ValueError('Second layer must not be an activation layer')
    super(CombinedLayer, self).__init__()
    self._activation_layer = activation_layer
    self._linear_layer = linear_layer

  @property
  def activation_layer(self):
    return self._activation_layer

  @property
  def linear_layer(self):
    return self._linear_layer

  @property
  def branches(self):
    return self._linear_layer.branches

  @property
  def input_node(self):
    return self._activation_layer.input_node

  @property
  def output_node(self):
    return self._linear_layer.output_node

  @property
  def reshape(self):
    return self._activation_layer.reshape

  def reshape_duals_backwards(self, dual_vars):
    if self.linear_layer.reshape:
      # There was a reshape prior to the linear layer.
      dual_vars = snt.BatchReshape(self.activation_layer.output_shape,
                                   preserve_dims=2)(dual_vars)
    return dual_vars


class CombinedLinearLayer(layers.VerifiableLayer):
  """Wraps linear layers, treating them as a single linear layer."""

  def __init__(self, first_layer, second_layer):
    """Constructor.

    Args:
      first_layer: `AffineLayer` or `CombinedLinearLayer`.
      second_layer: `AffineLayer` or `CombinedLinearLayer`.
    """
    super(CombinedLinearLayer, self).__init__()
    self._first_layer = first_layer
    self._second_layer = second_layer

  @property
  def first_layer(self):
    return self._first_layer

  @property
  def second_layer(self):
    return self._second_layer

  @property
  def branches(self):
    return self._second_layer.branches

  @property
  def input_node(self):
    return self._first_layer.input_node

  @property
  def output_node(self):
    return self._second_layer.output_node

  @property
  def reshape(self):
    return self._first_layer.reshape

  @property
  def is_activation(self):
    return False

  def reshape_duals_backwards(self, dual_vars):
    if self._second_layer.reshape:
      # There was a reshape prior to the second layer.
      dual_vars = snt.BatchReshape(self._first_layer.output_shape,
                                   preserve_dims=2)(dual_vars)
    return dual_vars

  def get_objective_weights(self, labels, target_strategy=None):
    # Obtain the final objective according to the second layer.
    next_obj_w, next_obj_b = self._second_layer.get_objective_weights(
        labels, target_strategy=target_strategy)

    # Un-flatten the final objective.
    num_classes = next_obj_w.shape[0].value
    batch_size = tf.shape(next_obj_w)[1]
    next_obj_w = tf.reshape(
        next_obj_w, [num_classes, batch_size] +
        list(self._first_layer.output_shape))

    # If this layer is w1_ij, b1_j
    # and the second layer's objective is w2_knj, b2_kn
    # then the overall objective is given by wr_kni, br_kn as follows:
    # w1_ij w2_knj  ->  wr_kni
    # b1_j w2_knj  +  b2_kn  ->  br_kn
    obj_w, obj_b = self._first_layer.backward_prop_batchnorm_bias(next_obj_w,
                                                                  next_obj_b)
    obj_b = obj_b + self._first_layer.backward_prop_bias(obj_w)
    obj_w = self._first_layer.backward_prop(obj_w)
    return layers.ObjectiveWeights(obj_w, obj_b)

  def forward_prop(self, x, apply_bias=False, w_fn=None):
    if (self._first_layer.batch_norm is not None or
        self._second_layer.batch_norm is not None):
      raise ValueError('Batch norm not supported.')
    x = self._first_layer.forward_prop(x, apply_bias=apply_bias, w_fn=w_fn)
    x = self._first_layer.reshape_duals_forwards(x, self._second_layer)
    x = self._second_layer.forward_prop(x, apply_bias=apply_bias, w_fn=w_fn)
    return x

  def backward_prop(self, y, w_fn=None):
    y = self._second_layer.backward_prop_batchnorm(y)
    y = self._second_layer.backward_prop(y, w_fn=w_fn)
    y = self.reshape_duals_backwards(y)
    y = self._first_layer.backward_prop_batchnorm(y)
    y = self._first_layer.backward_prop(y, w_fn=w_fn)
    return y

  def backward_prop_bias(self, y):
    bias = tf.zeros(tf.shape(y)[:2], dtype=y.dtype)
    y, bias = self._second_layer.backward_prop_batchnorm_bias(y, bias)
    bias = bias + self._second_layer.backward_prop_bias(y)
    y = self._second_layer.backward_prop(y)
    y = self.reshape_duals_backwards(y)
    y, bias = self._first_layer.backward_prop_batchnorm_bias(y, bias)
    bias = bias + self._first_layer.backward_prop_bias(y)
    return bias

  def flatten(self):
    if (self._first_layer.batch_norm is not None or
        self._second_layer.batch_norm is not None):
      raise ValueError('Batch norm not supported.')
    return tf.matmul(self._first_layer.flatten(),
                     self._second_layer.flatten())

  def backward_prop_batchnorm(self, y):
    return y

  def backward_prop_batchnorm_bias(self, y, bias):
    return y, bias


def combine_trailing_linear_layers(verifiable_layers):
  """If the network culminates in two or more linear layers, combines them.

  Args:
    verifiable_layers: List of `SingleVerifiableLayer`.

  Returns:
    List of `VerifiableLayer` in which trailing linear layers have been combined
      into one.

  Raises:
    ValueError: if an unsupported layer type or arrangement is encountered.
  """
  if not isinstance(verifiable_layers[-1], layers.Linear):
    raise ValueError('Final layers other than linear are not supported.')

  # Take a copy, to avoid mutating the input.
  final_layer = verifiable_layers[-1]
  verifiable_layers = verifiable_layers[:-1]

  while len(verifiable_layers) and not verifiable_layers[-1].is_activation:
    # Combine the last two layers.
    linear_layer = verifiable_layers.pop()
    final_layer = CombinedLinearLayer(linear_layer, final_layer)
  verifiable_layers.append(final_layer)

  return verifiable_layers

