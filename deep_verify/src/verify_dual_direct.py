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

"""Graph construction for dual verification: Lagrangian calculation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class DualVerification(snt.AbstractModule):
  """Module to represent a network's Lagrangian, in terms of dual variables."""

  def __init__(self, verification_strategy,
               verifiable_layers,
               target_strategy=None,
               get_dual_variable=tf.get_variable,
               name='dual_verification'):
    """Initialises the dual verification module.

    Args:
      verification_strategy: strategy object defining the dual verification
        formulation, including what dual variables exist for each layer.
      verifiable_layers: List of `VerifiableLayer` objects specifying
        linear layers and non-linear activation functions.
      target_strategy: target_objective_strategy object defining the objective
        to optimize for the Lagrangian, default set to None, in which case we
        will use the standard verification objective.
      get_dual_variable: Function(name, shape, dtype) returning a dual variable.
        It will be invoked by keyword arguments, so its arguments must be named
        as given here, but may occur in any order.
      name: Optional name for the module, defaulting to 'dual_verification'.
    """
    super(DualVerification, self).__init__(name=name)
    self._verification_strategy = verification_strategy
    self._verifiable_layers = verifiable_layers
    self._target_strategy = target_strategy
    self._get_dual_variable = get_dual_variable

  def _build(self, labels, num_batches, current_batch,
             margin=0.,
             objective_computation_config=None):
    """Sets the up dual objective for the given network.

    Dual variables are allocated for the entire dataset, covering all batches
    as specified by `num_batches`. The dual objective accesses a slice of the
    the dual variables specified by `current_batch`.

    Args:
      labels: 1D integer tensor of shape (batch_size) of labels for each
        input example.
      num_batches: Total number of batches in the dataset.
      current_batch: 0D integer tensor containing index of current batch.
      margin: Dual objective values for correct class will be forced to
        `-margin`, thus disregarding large negative bounds when maximising.
      objective_computation_config: Additional parameters for dual obj.

    Returns:
      2D tensor of shape (num_targets, batch_size) containing dual objective
        values for each (class, example).
    """
    # Dual variable generation across all batches.
    batch_size = labels.shape[0]
    batch_lo = current_batch * batch_size
    batch_hi = batch_lo + batch_size

    def dual_var_getter(name, shape, dtype):
      """Creates a trainable tf.Variable for each dual variables."""
      dual_var = self._get_dual_variable(name=name,
                                         dtype=dtype,
                                         shape=(shape[:1] +
                                                [num_batches * batch_size] +
                                                shape[2:]))
      # Return directly the tf.Variable if possible.
      if num_batches == 1:
        return dual_var
      # Select correct slice of dual variables for current batch.
      sliced = dual_var[:, batch_lo:batch_hi]
      sliced.set_shape(shape)
      return sliced

    (dual_obj, self._dual_var_lists, self._project_duals_op,
     self._supporting_ops) = (
         self._verification_strategy.create_duals_and_build_objective(
             self._verifiable_layers,
             labels,
             dual_var_getter,
             margin=margin,
             target_strategy=self._target_strategy,
             objective_computation_config=objective_computation_config))
    return dual_obj

  @property
  def dual_var_lists(self):
    """TensorFlow variables for all dual variables."""
    self._ensure_is_connected()
    return self._dual_var_lists

  @property
  def project_duals_op(self):
    """TensorFlow operation to project all dual variables to their bounds."""
    self._ensure_is_connected()
    return self._project_duals_op

  @property
  def init_duals_op(self):
    """TensorFlow operation to initialize dual variables."""
    return self.supporting_ops['init']

  @property
  def supporting_ops(self):
    """Additional TF ops (e.g. initialization) for the dual variables."""
    self._ensure_is_connected()
    return self._supporting_ops

  def dual_variables_by_name(self, names):
    """Get dual variables by name."""
    return _dual_variables_by_name(names, self.dual_var_lists)


def _dual_variables_by_name(names, dual_var_lists):
  dual_vars = []
  for dual_var_list in dual_var_lists:
    for child_dual_var_lists in dual_var_list[:-1]:
      dual_vars.extend(_dual_variables_by_name(names, child_dual_var_lists))
    dual = dual_var_list[-1]
    if dual is not None:
      dual_vars.extend([dual[name] for name in names if name in dual])
  return dual_vars
