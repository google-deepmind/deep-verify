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

"""Fast-Lin bound calculation for common neural network layers.

The Fast-Lin algorithm expresses lower and upper bounds of each layer of
a neural network as a symbolic linear expression in the input neurons,
relaxing the ReLU layers to retain linearity at the expense of tightness.

Reference: "Towards Fast Computation of Certified Robustness for ReLU Networks",
https://arxiv.org/pdf/1804.09699.pdf.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.bounds import layer_bounds
import interval_bound_propagation as ibp


class FastlinBoundPropagation(layer_bounds.BoundPropagation):
  """Method for propagating symbolic bounds in multiple passes."""

  def __init__(self, num_rounds=1, best_with_naive=False):
    super(FastlinBoundPropagation, self).__init__()
    self._num_rounds = num_rounds
    self._best_with_naive = best_with_naive

  def propagate_bounds(self, network, in_bounds):
    if self._best_with_naive:
      # Initial round of interval bound propagation.
      super(FastlinBoundPropagation, self).propagate_bounds(network, in_bounds)

    for _ in range(self._num_rounds):
      # Construct symbolic bounds and propagate them.
      super(FastlinBoundPropagation, self).propagate_bounds(
          network, ibp.RelativeSymbolicBounds.convert(in_bounds))
