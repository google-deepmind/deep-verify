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

"""Defines a target objective specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class TargetObjective(object):
  """Specifies the target objective to be minimized."""
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
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

  @abc.abstractmethod
  def filter_correct_class(self, dual_obj, labels, margin):
    """Filters out the objective when the target class contains the true label.

    Args:
      dual_obj: 2D tensor of shape (num_targets, batch_size) containing
        dual objectives.
      labels: 1D tensor of shape (batch_size) containing the labels for each
        example in the batch.
      margin: Dual objective values for correct class will be forced to
        `-margin`, thus disregarding large negative bounds when maximising.

    Returns:
     2D tensor of shape (num_classes, batch_size) containing the corrected dual
     objective values for each (class, example).
    """

  @abc.abstractproperty
  def num_targets(self):
    """Returns the number of targets in the objective."""



