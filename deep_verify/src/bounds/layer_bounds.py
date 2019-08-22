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

"""Bound calculation interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc


class BoundPropagation(object):
  """Method for propagating bounds through the layers."""
  __metaclass__ = abc.ABCMeta

  def propagate_bounds(self, network, in_bounds):
    """Calculates bounds on each layer.

    Args:
      network: `auto_verifier.NetworkBuilder` specifying network graph.
      in_bounds: Bounds for the network inputs.
    """
    network.propagate_bounds(in_bounds)

