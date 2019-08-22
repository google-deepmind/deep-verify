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

"""Tests for dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deep_verify.src import common
from deep_verify.src.bounds import naive_bounds
from deep_verify.src.formulations.standard import standard_layer_calcs
from deep_verify.src.formulations.standard import verify_dual_standard
from deep_verify.tests.formulations import verify_dual_base_test
import sonnet as snt
import tensorflow as tf


class VerifyDualStandardTest(verify_dual_base_test.DualFormulationTest):

  def _verification_strategy(self):
    return verify_dual_standard.StandardDualFormulation(use_reduced=False)

  @parameterized.named_parameters(('linear', 'linear'),
                                  ('conv', 'conv'),
                                  ('conv_batchnorm', 'conv_batchnorm'),
                                  ('avgpool', 'avgpool'),
                                  ('avgpool_linear', 'avgpool_linear'))
  def test_run(self, model):
    self._apply_verification(model)

  def test_calc_linear(self):
    image_data = self._image_data()
    net = self._network('linear')
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    dual_obj, dual_var_lists = self._build_objective(net, input_bounds,
                                                     image_data.label)

    # Explicitly build the expected TensorFlow graph for calculating objective.
    (linear_0,
     relu_1,  # pylint:disable=unused-variable
     linear_obj) = self._verifiable_layer_builder(net).build_layers()
    (mu_0,), _ = dual_var_lists

    # Expected input bounds for each layer.
    linear_0_lb, linear_0_ub = self._expected_input_bounds(image_data.image, .1)
    linear_0_lb = snt.BatchFlatten()(linear_0_lb)
    linear_0_ub = snt.BatchFlatten()(linear_0_ub)
    relu_1_lb, relu_1_ub = naive_bounds.linear_bounds(
        linear_0.module.w, linear_0.module.b,
        linear_0_lb, linear_0_ub)

    # Expected objective value.
    objective = 0
    act_coeffs_0 = -tf.tensordot(mu_0, tf.transpose(linear_0.module.w), axes=1)
    obj_0 = -tf.tensordot(mu_0, linear_0.module.b, axes=1)
    objective += standard_layer_calcs.linear_dual_objective(
        None, act_coeffs_0, obj_0, linear_0_lb, linear_0_ub)
    objective_w, objective_b = common.targeted_objective(
        linear_obj.module.w, linear_obj.module.b, image_data.label)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_0, -objective_w,
        relu_1_lb, relu_1_ub)
    objective += objective_b

    self._assert_dual_objective_close(objective, dual_obj, image_data)

  def test_calc_conv(self):
    image_data = self._image_data()
    net = self._network('conv')
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    dual_obj, dual_var_lists = self._build_objective(net, input_bounds,
                                                     image_data.label)

    # Explicitly build the expected TensorFlow graph for calculating objective.
    (conv2d_0,
     relu_1,  # pylint:disable=unused-variable
     linear_2,
     relu_3,  # pylint:disable=unused-variable
     linear_obj) = self._verifiable_layer_builder(net).build_layers()
    (mu_0,), (lam_1,), (mu_2,), _ = dual_var_lists

    # Expected input bounds for each layer.
    conv2d_0_lb, conv2d_0_ub = self._expected_input_bounds(image_data.image, .1)
    relu_1_lb, relu_1_ub = naive_bounds.conv2d_bounds(
        conv2d_0.module.w, conv2d_0.module.b, 'VALID', (1, 1),
        conv2d_0_lb, conv2d_0_ub)
    linear_2_lb = snt.BatchFlatten()(tf.nn.relu(relu_1_lb))
    linear_2_ub = snt.BatchFlatten()(tf.nn.relu(relu_1_ub))
    relu_3_lb, relu_3_ub = naive_bounds.linear_bounds(
        linear_2.module.w, linear_2.module.b,
        linear_2_lb, linear_2_ub)

    # Expected objective value.
    objective = 0
    act_coeffs_0 = -common.conv_transpose(mu_0, conv2d_0.module.w,
                                          conv2d_0.input_shape,
                                          'VALID', (1, 1))
    obj_0 = -tf.reduce_sum(mu_0 * conv2d_0.module.b, axis=(2, 3, 4))
    objective += standard_layer_calcs.linear_dual_objective(
        None, act_coeffs_0, obj_0, conv2d_0_lb, conv2d_0_ub)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_0, lam_1,
        relu_1_lb, relu_1_ub)
    act_coeffs_2 = -tf.tensordot(mu_2, tf.transpose(linear_2.module.w), axes=1)
    obj_2 = -tf.tensordot(mu_2, linear_2.module.b, axes=1)
    objective += standard_layer_calcs.linear_dual_objective(
        snt.BatchFlatten(preserve_dims=2)(lam_1),
        act_coeffs_2, obj_2, linear_2_lb, linear_2_ub)
    objective_w, objective_b = common.targeted_objective(
        linear_obj.module.w, linear_obj.module.b, image_data.label)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_2, -objective_w,
        relu_3_lb, relu_3_ub)
    objective += objective_b

    self._assert_dual_objective_close(objective, dual_obj, image_data)

  def test_calc_conv_batchnorm(self):
    image_data = self._image_data()
    net = self._network('conv_batchnorm')
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    dual_obj, dual_var_lists = self._build_objective(net, input_bounds,
                                                     image_data.label)

    # Explicitly build the expected TensorFlow graph for calculating objective.
    (conv2d_0,
     relu_1,  # pylint:disable=unused-variable
     linear_2,
     relu_3,  # pylint:disable=unused-variable
     linear_obj) = self._verifiable_layer_builder(net).build_layers()
    (mu_0,), (lam_1,), (mu_2,), _ = dual_var_lists

    # Expected input bounds for each layer.
    conv2d_0_lb, conv2d_0_ub = self._expected_input_bounds(image_data.image, .1)
    conv2d_0_w, conv2d_0_b = common.combine_with_batchnorm(
        conv2d_0.module.w, None, conv2d_0.batch_norm)
    relu_1_lb, relu_1_ub = naive_bounds.conv2d_bounds(
        conv2d_0_w, conv2d_0_b, 'VALID', (1, 1),
        conv2d_0_lb, conv2d_0_ub)
    linear_2_lb = snt.BatchFlatten()(tf.nn.relu(relu_1_lb))
    linear_2_ub = snt.BatchFlatten()(tf.nn.relu(relu_1_ub))
    linear_2_w, linear_2_b = common.combine_with_batchnorm(
        linear_2.module.w, None, linear_2.batch_norm)
    relu_3_lb, relu_3_ub = naive_bounds.linear_bounds(
        linear_2_w, linear_2_b,
        linear_2_lb, linear_2_ub)

    # Expected objective value.
    objective = 0
    act_coeffs_0 = -common.conv_transpose(mu_0, conv2d_0_w,
                                          conv2d_0.input_shape,
                                          'VALID', (1, 1))
    obj_0 = -tf.reduce_sum(mu_0 * conv2d_0_b, axis=(2, 3, 4))
    objective += standard_layer_calcs.linear_dual_objective(
        None, act_coeffs_0, obj_0, conv2d_0_lb, conv2d_0_ub)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_0, lam_1,
        relu_1_lb, relu_1_ub)
    act_coeffs_2 = -tf.tensordot(mu_2, tf.transpose(linear_2_w), axes=1)
    obj_2 = -tf.tensordot(mu_2, linear_2_b, axes=1)
    objective += standard_layer_calcs.linear_dual_objective(
        snt.BatchFlatten(preserve_dims=2)(lam_1),
        act_coeffs_2, obj_2, linear_2_lb, linear_2_ub)
    objective_w, objective_b = common.targeted_objective(
        linear_obj.module.w, linear_obj.module.b, image_data.label)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_2, -objective_w,
        relu_3_lb, relu_3_ub)
    objective += objective_b

    self._assert_dual_objective_close(objective, dual_obj, image_data)

  def test_calc_avgpool(self):
    image_data = self._image_data()
    net = self._network('avgpool')
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    dual_obj, dual_var_lists = self._build_objective(net, input_bounds,
                                                     image_data.label)

    # Explicitly build the expected TensorFlow graph for calculating objective.
    (conv2d_0,
     relu_1,  # pylint:disable=unused-variable
     avgpool_2,
     relu_3,  # pylint:disable=unused-variable
     linear_obj) = self._verifiable_layer_builder(net).build_layers()
    (mu_0,), (lam_1,), (mu_2,), _ = dual_var_lists

    # Expected input bounds for each layer.
    conv2d_0_lb, conv2d_0_ub = self._expected_input_bounds(image_data.image, .1)
    relu_1_lb, relu_1_ub = naive_bounds.conv2d_bounds(
        conv2d_0.module.w, conv2d_0.module.b, 'SAME', (1, 1),
        conv2d_0_lb, conv2d_0_ub)
    avgpool_2_lb = tf.nn.relu(relu_1_lb)
    avgpool_2_ub = tf.nn.relu(relu_1_ub)
    relu_3_lb = tf.nn.avg_pool(avgpool_2_lb, ksize=[2, 2],
                               padding='VALID', strides=(1, 1))
    relu_3_ub = tf.nn.avg_pool(avgpool_2_ub, ksize=[2, 2],
                               padding='VALID', strides=(1, 1))

    # Expected objective value.
    objective = 0
    act_coeffs_0 = -common.conv_transpose(mu_0, conv2d_0.module.w,
                                          conv2d_0.input_shape,
                                          'SAME', (1, 1))
    obj_0 = -tf.reduce_sum(mu_0 * conv2d_0.module.b, axis=(2, 3, 4))
    objective += standard_layer_calcs.linear_dual_objective(
        None, act_coeffs_0, obj_0, conv2d_0_lb, conv2d_0_ub)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_0, lam_1,
        relu_1_lb, relu_1_ub)
    act_coeffs_2 = -common.avgpool_transpose(
        mu_2, result_shape=relu_1.output_shape,
        kernel_shape=(2, 2), strides=(1, 1))
    objective += standard_layer_calcs.linear_dual_objective(
        lam_1, act_coeffs_2, 0., avgpool_2_lb, avgpool_2_ub)
    objective_w, objective_b = common.targeted_objective(
        linear_obj.module.w, linear_obj.module.b, image_data.label)
    shaped_objective_w = tf.reshape(objective_w,
                                    [self._num_classes(), self._batch_size()] +
                                    avgpool_2.output_shape)
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_2, -shaped_objective_w,
        relu_3_lb, relu_3_ub)
    objective += objective_b

    self._assert_dual_objective_close(objective, dual_obj, image_data)

  def test_calc_avgpool_linear(self):
    image_data = self._image_data()
    net = self._network('avgpool_linear')
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    dual_obj, dual_var_lists = self._build_objective(net, input_bounds,
                                                     image_data.label)

    # Explicitly build the expected TensorFlow graph for calculating objective.
    (conv2d_0,
     relu_1,
     avgpool_2,
     linear_obj) = self._verifiable_layer_builder(net).build_layers()
    (mu_0,), _ = dual_var_lists

    # Expected input bounds for each layer.
    conv2d_0_lb, conv2d_0_ub = self._expected_input_bounds(image_data.image, .1)
    relu_1_lb, relu_1_ub = naive_bounds.conv2d_bounds(
        conv2d_0.module.w, conv2d_0.module.b, 'SAME', (1, 1),
        conv2d_0_lb, conv2d_0_ub)

    # Expected objective value.
    objective = 0
    act_coeffs_0 = -common.conv_transpose(mu_0, conv2d_0.module.w,
                                          conv2d_0.input_shape,
                                          'SAME', (1, 1))
    obj_0 = -tf.reduce_sum(mu_0 * conv2d_0.module.b, axis=(2, 3, 4))
    objective += standard_layer_calcs.linear_dual_objective(
        None, act_coeffs_0, obj_0, conv2d_0_lb, conv2d_0_ub)
    objective_w, objective_b = common.targeted_objective(
        linear_obj.module.w, linear_obj.module.b, image_data.label)
    combined_objective_w = common.avgpool_transpose(
        tf.reshape(objective_w,
                   [self._num_classes(), self._batch_size()] +
                   avgpool_2.output_shape),
        relu_1.output_shape,
        kernel_shape=(2, 2), strides=(1, 1))
    objective += standard_layer_calcs.activation_layer_dual_objective(
        tf.nn.relu,
        mu_0, -combined_objective_w,
        relu_1_lb, relu_1_ub)
    objective += objective_b

    self._assert_dual_objective_close(objective, dual_obj, image_data)


if __name__ == '__main__':
  tf.test.main()
