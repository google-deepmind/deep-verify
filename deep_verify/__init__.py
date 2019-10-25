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

"""Library to verify robustness of neural networks using dual methods.

For more details see paper: "A Dual Approach to Scalable Verification
of Deep Networks.", https://arxiv.org/abs/1803.06567.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src import common

from deep_verify.src.auto_verifier import NotVerifiableError
from deep_verify.src.auto_verifier import VerifiableLayerBuilder

from deep_verify.src.bounds import naive_bounds
from deep_verify.src.bounds.fastlin_bounds import FastlinBoundPropagation
from deep_verify.src.bounds.layer_bounds import BoundPropagation
from deep_verify.src.bounds.naive_bounds import input_bounds
from deep_verify.src.bounds.naive_bounds import NaiveBoundPropagation

from deep_verify.src.common import with_explicit_update

from deep_verify.src.formulations.semidefinite import gram_calcs
from deep_verify.src.formulations.semidefinite.verify_dual_semidefinite import SemidefiniteDualFormulation
from deep_verify.src.formulations.standard import standard_layer_calcs
from deep_verify.src.formulations.standard.verify_dual_standard import StandardDualFormulation
from deep_verify.src.formulations.verify_dual_base import build_dual_vars
from deep_verify.src.formulations.verify_dual_base import build_project_duals_op
from deep_verify.src.formulations.verify_dual_base import DualFormulation

from deep_verify.src.layers.layers import Activation
from deep_verify.src.layers.layers import AffineLayer
from deep_verify.src.layers.layers import AvgPool
from deep_verify.src.layers.layers import Conv
from deep_verify.src.layers.layers import CustomOp
from deep_verify.src.layers.layers import Linear
from deep_verify.src.layers.layers import MaxPool
from deep_verify.src.layers.layers import SingleVerifiableLayer
from deep_verify.src.layers.layers import VerifiableLayer
from deep_verify.src.layers.layers_combined import combine_trailing_linear_layers
from deep_verify.src.layers.layers_combined import CombinedLayer
from deep_verify.src.layers.layers_combined import CombinedLinearLayer

from deep_verify.src.specifications.target_objective_base import TargetObjective
from deep_verify.src.specifications.target_objective_standard import StandardTargetObjective

from deep_verify.src.verify_dual_direct import DualVerification
