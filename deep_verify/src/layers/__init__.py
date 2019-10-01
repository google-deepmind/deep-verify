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

"""Layer wrappers for verification, allowing dual variables to be defined.

Instances of `layers.VerifiableLayer` are created by the graph analyser
(`auto_verifier.VerifiableLayerBuilder`) to wrap a single layer of the network.

Individual dual formulations may choose to further combine these into coarser
blocks (e.g. activation+linear) that form meaningful units for the verification
method.

In general, any verifiable layer will declare some dual variables to be
associated with that layer. Optimising with respect to the duals (for each
input example) will give the greatest chance of finding a verifiability proof.
By default, a layer will have a single dual variable for each output neuron,
corresponding to the Lagrange multiplier for the constrant that the neuron's
value matches the corresponding input value on the next layer. Other
non-standard formulations can define custom layer groupings specifying
their own collections of dual variables.

For more details see paper: "A Dual Approach to Scalable Verification
of Deep Networks.", https://arxiv.org/abs/1803.06567.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
