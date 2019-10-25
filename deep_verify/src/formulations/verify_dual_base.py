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

"""Defines a strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import logging
from deep_verify.src.specifications import target_objective_standard
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class DualFormulation(object):
  """Specifies which dual variables exist for which layers."""

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  def create_duals_and_build_objective(
      self, verifiable_layers,
      labels,
      dual_var_getter,
      margin=0.,
      target_strategy=None,
      objective_computation_config=None):
    """Sets the up dual objective.

    Args:
      verifiable_layers: List of `VerifiableLayer` objects specifying layers
        that give rise to dual variables.
      labels: 1D integer tensor of shape (batch_size) of labels for each
        input example.
      dual_var_getter: Function(name, shape, dtype) returning a dual variable.
      margin: Dual objective values for correct class will be forced to
        `-margin`, thus disregarding large negative bounds when maximising.
      target_strategy: target_objective_strategy object defining the objective
        to optimize for the Lagrangian, default set to None, in which case we
        will use the standard verification objective.
      objective_computation_config: Additional params to dual obj calculation.

    Returns:
      dual_objective: 2D tensor of shape (num_classes, batch_size) containing
        dual objective values for each (class, example).
      dual_var_lists: Nested list of 3D tensors of shape
        (num_classes, batch_size, layer_size) and 5D tensors of shape
        (num_classes, batch_size, height, width, channels)
        containing Lagrange multipliers for the layers' calculations.
        This has the same length as `verifiable_layers`, and typically each
        entry is a singleton list with the dual variable for that layer.
        ResNet blocks' entries have instead the structure
        [[left-sub-duals], [right-sub-duals], overall-dual].
      init_duals_op: TensorFlow operation to initialise all dual variables.
      project_duals_op: TensorFlow operation to project all dual variables
        to their bounds.
      supporting_ops: Dictionary of additional ops (e.g. 'init') for
        manipulating the dual variables.
        The set of keys is implementation-dependent, according to what the
        particular formulation supports.
    """
    target_strategy = target_strategy or (
        target_objective_standard.StandardTargetObjective(
            verifiable_layers[-1].output_shape[-1]))
    batch_size = labels.shape[0]
    dtype = verifiable_layers[-1].output_bounds.lower.dtype

    # Obtain TensorFlow variables for all dual variables.
    def dual_var_getter_full(unused_layer, name, dual_shape):
      return dual_var_getter(name=name,
                             dtype=dtype,
                             shape=([target_strategy.num_targets, batch_size] +
                                    list(dual_shape)))
    dual_var_lists = build_dual_vars(verifiable_layers,
                                     dual_var_getter_full)
    # Create a 'project' TensorFlow op to clamp the dual vars to their bounds.
    project_duals_op = build_project_duals_op(verifiable_layers, dual_var_lists)

    # Calculate the dual objective.
    dual_obj = self.dual_objective(
        verifiable_layers, labels, dual_var_lists,
        target_strategy=target_strategy,
        objective_computation_config=objective_computation_config)

    # Filter out cases in which the target class is the correct class.
    dual_obj = target_strategy.filter_correct_class(
        dual_obj, tf.cast(labels, tf.int32), margin=margin)

    # Build additional ops to manipulate the variables (e.g. initialisation).
    supporting_ops = self.create_supporting_ops(
        verifiable_layers, labels, dual_var_lists,
        target_strategy=target_strategy,
        objective_computation_config=objective_computation_config)
    return dual_obj, dual_var_lists, project_duals_op, supporting_ops

  def create_supporting_ops(self, verifiable_layers, labels, dual_var_lists,
                            target_strategy=None,
                            objective_computation_config=None):
    """Creates additional ops (e.g. initialization) for the dual variables.

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
      Dictionary containing additional ops.
      The set of keys is implementation-dependent, according to what a
      particular formulation supports.
    """
    del verifiable_layers, labels, dual_var_lists, target_strategy
    del objective_computation_config
    return {'init': tf.no_op()}


def build_dual_vars(verifiable_layers, dual_var_getter):
  """Creates dual variable list for the given layers.

  Args:
    verifiable_layers: Layers for which dual variables should be generated.
    dual_var_getter: Function(layer, name, shape) returning a dual variable.

  Returns:
    Nested list of dual variable tensors, one list for each layer.
    Each layer typically has a singleton list with its dual variable.
    ResNet blocks will have instead the structure
    [[left-sub-duals], [right-sub-duals], overall-dual].
  """
  return [_dual_var_list_for_layer(dual_var_getter, layer, 'dual_{}'.format(i))
          for i, layer in enumerate(verifiable_layers)]


def _dual_var_for_layer(dual_var_getter, layer, name, shape):
  """Creates a dual variable for the given layer shape.

  Args:
    dual_var_getter: Function(name, shape) returning a dual variable.
    layer: Layer for which dual variables should be generated.
    name: Name to use for dual variable.
    shape: Shape of the dual variable, or a possibly nested dict of shapes.

  Returns:
    Dual variable tensors of the given shape, or a possibly nested dict of
    such tensors according to the structure of `shape`.
  """
  if isinstance(shape, dict):
    return {k: _dual_var_for_layer(dual_var_getter, layer, name + '_' + k, v)
            for k, v in shape.items()}
  else:
    return dual_var_getter(layer, name, shape)


def _dual_var_list_for_layer(dual_var_getter, layer, name):
  """Creates dual variable list for the given layer.

  Args:
    dual_var_getter: Function(name, shape) returning a dual variable.
    layer: Layer for which dual variables should be generated.
    name: Name to use for dual variable.

  Returns:
    List of dual variable tensors. Entries may be `None`.
    This is typically a singleton list with the dual variable for the layer.
    ResNet blocks will have instead the structure
    [[left-sub-duals], [right-sub-duals], overall-dual].
  """
  # Dual variable may not be required.
  if layer.dual_shape() is None:
    dual_var = None
  else:
    dual_var = _dual_var_for_layer(dual_var_getter, layer, name,
                                   layer.dual_shape())

  dual_var_list = []

  # Precede with dual vars for branches, if it's a ResNet block.
  for branch_name, branch_layers in layer.branches:
    child_dual_var_lists = [
        _dual_var_list_for_layer(dual_var_getter, sublayer,
                                 '{}_{}_{}'.format(name, branch_name, i))
        for i, sublayer in enumerate(branch_layers)]
    dual_var_list.append(child_dual_var_lists)

  dual_var_list.append(dual_var)
  return dual_var_list


def build_project_duals_op(verifiable_layers, dual_var_lists):
  """Projects duals into their regional constraints.

  Args:
    verifiable_layers: List of `VerifiableLayer` objects specifying layers
      that give rise to dual variables.
    dual_var_lists: Nested list of TensorFlow variables containing Lagrange
      multipliers for the layers' calculations.
      This has the same length as `verifiable_layers`, and typically each
      entry is a singleton list with the dual variable for that layer.
      ResNet blocks' entries have instead the structure
      [[left-sub-duals], [right-sub-duals], overall-dual].

  Returns:
    TensorFlow operation that updates the variables in `dual_var_lists`,
    clamping them to their admissible ranges (where relevant).
  """
  return tf.group(
      *[_project_duals_op_for_layer(layer, dual_var_list)
        for layer, dual_var_list in zip(verifiable_layers, dual_var_lists)])


def _project_duals_op_for_layer(layer, dual_var_list):
  """Returns op that updates duals for a single layer."""
  # Dual variable may not be required.
  if layer.dual_shape() is None:
    project_op = tf.no_op()
  else:
    try:
      project_op = layer.project_duals_op(dual_var_list[-1])
    except (ValueError, AttributeError):
      logging.warn('Cannot create projection.')
      # Use an un-fed placeholder to force an error at graph execution time.
      with tf.control_dependencies([tf.placeholder(dtype=tf.float32,
                                                   shape=(),
                                                   name='cannot_project')]):
        project_op = tf.no_op()

  project_ops = []

  # Precede with dual vars for branches, if it's a ResNet block.
  for (_, branch_layers), child_dual_var_lists in zip(layer.branches,
                                                      dual_var_list[:-1]):
    project_ops.append(
        build_project_duals_op(branch_layers, child_dual_var_lists))

  project_ops.append(project_op)
  return tf.group(*project_ops)
