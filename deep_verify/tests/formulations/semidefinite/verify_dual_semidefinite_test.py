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

"""Tests for semidefinite formulation of dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deep_verify.src.formulations.semidefinite import verify_dual_semidefinite
from deep_verify.tests.formulations import verify_dual_base_test
import tensorflow as tf


class SemidefiniteDualFormulationTest(
    verify_dual_base_test.DualFormulationTest):

  def _verification_strategy(self):
    return verify_dual_semidefinite.SemidefiniteDualFormulation()

  @parameterized.named_parameters(('linear', 'linear'),
                                  ('conv', 'conv'))
  def test_semidefinite(self, model):
    self._apply_verification(model)

  @parameterized.named_parameters(('linear', 'linear'),
                                  ('conv', 'conv'))
  def test_semidefinite_weak(self, model):
    self._apply_verification(model, {'verify_option': 'weak'})

  @parameterized.named_parameters(('linear', 'linear'),
                                  ('conv', 'conv'))
  def test_semidefinite_strong(self, model):
    self._apply_verification(model, {'verify_option': 'strong'})

  @parameterized.named_parameters(('linear', 'linear'),
                                  ('conv', 'conv'))
  def test_semidefinite_strong_approx(self, model):
    self._apply_verification(model, {'verify_option': 'strong_approx'})


if __name__ == '__main__':
  tf.test.main()
