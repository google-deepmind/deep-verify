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

"""Defines standard 'correct class' target objective specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.specifications import target_objective_base
import tensorflow as tf


class StandardTargetObjective(target_objective_base.TargetObjective):
  """Specifies the target objective to be minimized."""

  def __init__(self, num_classes):
    super(StandardTargetObjective, self).__init__()
    self._num_targets = num_classes

  def target_objective(self, final_w, final_b, labels):
    """Computes the true objective which is being optimized.

    Args:
      final_w: 2D tensor of shape (last_hidden_layer_size, num_classes)
        containing the weights for the final linear layer.
      final_b: 1D tensor of shape (num_classes) containing the biases for the
        final hidden layer.
      labels: 1D integer tensor of shape (batch_size) of labels for each
        input example.

    Returns:
      obj_w: Tensor of shape (num_targets, batch_size, last_hidden_layer_size)
        containing weights (to use in place of final linear layer weights)
        for targeted attacks.
      obj_b: Tensor of shape (num_targets, batch_size) containing bias
        (to use in place of final linear layer biases) for targeted attacks.
    """
    # Elide objective with final linear layer.
    final_wt = tf.transpose(final_w)

    obj_w = (tf.expand_dims(final_wt, axis=1)
             - tf.gather(final_wt, labels, axis=0))
    obj_b = tf.expand_dims(final_b, axis=1) - tf.gather(final_b, labels, axis=0)
    return obj_w, obj_b

  def filter_correct_class(self, dual_obj, labels, margin=0.):
    """Filters out the objective when the target class contains the true label.

    Args:
      dual_obj: 2D tensor of shape (num_targets, batch_size) containing
        dual objectives.
      labels: 1D tensor of shape (batch_size) containing the labels for each
        example in the batch.
      margin: Dual objective values for correct class will be forced to
        `-margin`, thus disregarding large negative bounds when maximising. By
        default this is set to 0.

    Returns:
     2D tensor of shape (num_classes, batch_size) containing the corrected dual
     objective values for each (class, example).
    """
    neq = self.neq(labels)
    dual_obj = tf.where(neq, dual_obj, -margin * tf.ones_like(dual_obj))
    return dual_obj

  def neq(self, labels):
    assert hasattr(labels, 'dtype')
    targets_to_filter = tf.expand_dims(
        tf.range(self._num_targets, dtype=labels.dtype), axis=1)
    return tf.not_equal(targets_to_filter, labels)

  @property
  def num_targets(self):
    return self._num_targets
