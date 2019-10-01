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

"""Dual formulations of verification.

This module defines the interface for layer-wise formulations of verifiable
robustness that involve optimising dual variables (such as Lagrange multipliers)
associated with the layers.

For more details see paper: "A Dual Approach to Scalable Verification
of Deep Networks.", https://arxiv.org/abs/1803.06567.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
