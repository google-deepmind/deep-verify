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

"""Tests for dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deep_verify.src import common
from deep_verify.src.formulations.standard import standard_layer_calcs
import interval_bound_propagation as ibp
import numpy as np
import sonnet as snt
import tensorflow as tf


class StandardLayerCalcsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_linear_layer_dual_objective_shape(self, dtype):
    num_classes = 3
    batch_size = 11
    input_size = 7
    output_size = 5

    w = tf.placeholder(dtype=dtype, shape=(input_size, output_size))
    b = tf.placeholder(dtype=dtype, shape=(output_size,))
    lam_in = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, input_size))
    mu_out = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, output_size))
    lb = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))
    ub = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))

    activation_coeffs = -tf.tensordot(mu_out, tf.transpose(w), axes=1)
    dual_obj_bias = -tf.tensordot(mu_out, b, axes=1)
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    self.assertEqual(dtype, dual_obj.dtype)
    self.assertEqual((num_classes, batch_size), dual_obj.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_linear_layer_dual_objective(self, dtype, tol):
    w = tf.constant([[1.0, 2.0, 3.0], [4.0, -5.0, -6.0]], dtype=dtype)
    b = tf.constant([0.1, 0.2, 0.3], dtype=dtype)
    lb = tf.constant([[-1.0, -1.0]], dtype=dtype)
    ub = tf.constant([[1.0, 1.0]], dtype=dtype)

    lam_in = tf.constant([[[-.01, -.02]]], dtype=dtype)
    mu_out = tf.constant([[[30.0, 40.0, 50.0]]], dtype=dtype)
    # Activation coefficients: -.01 - 260, and -.02 + 380

    activation_coeffs = -tf.tensordot(mu_out, tf.transpose(w), axes=1)
    dual_obj_bias = -tf.tensordot(mu_out, b, axes=1)
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    dual_obj_exp = np.array([[(.01 + 260.0) + (-.02 + 380.0) - 26.0]])

    with self.test_session() as session:
      dual_obj_act = session.run(dual_obj)
      self.assertAllClose(dual_obj_exp, dual_obj_act, atol=tol, rtol=tol)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_conv2d_layer_dual_objective_shape(self, dtype):
    num_classes = 6
    batch_size = 23
    input_height = 17
    input_width = 7
    kernel_height = 3
    kernel_width = 4
    input_channels = 3
    output_channels = 5
    padding = 'VALID'
    strides = (2, 1)

    # Output dimensions, based on convolution settings.
    output_height = 8
    output_width = 4

    w = tf.placeholder(dtype=dtype, shape=(
        kernel_height, kernel_width, input_channels, output_channels))
    b = tf.placeholder(dtype=dtype, shape=(output_channels,))
    lam_in = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, input_height, input_width, input_channels))
    mu_out = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, output_height, output_width, output_channels))
    lb = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    ub = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))

    activation_coeffs = -common.conv_transpose(mu_out, w,
                                               lb.shape[1:].as_list(),
                                               padding, strides)
    dual_obj_bias = -tf.reduce_sum(mu_out * b, axis=(2, 3, 4))
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    self.assertEqual(dtype, dual_obj.dtype)
    self.assertEqual((num_classes, batch_size), dual_obj.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_conv2d_layer_dual_objective(self, dtype, tol):
    num_classes = 5
    batch_size = 53
    input_height = 17
    input_width = 7
    kernel_height = 3
    kernel_width = 4
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2, 1)

    # Output dimensions, based on convolution settings.
    output_height = 8
    output_width = 4

    w = tf.random_normal(dtype=dtype, shape=(
        kernel_height, kernel_width, input_channels, output_channels))
    b = tf.random_normal(dtype=dtype, shape=(output_channels,))
    lam_in = tf.random_normal(dtype=dtype, shape=(
        num_classes, batch_size, input_height, input_width, input_channels))
    mu_out = tf.random_normal(dtype=dtype, shape=(
        num_classes, batch_size, output_height, output_width, output_channels))
    lb = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    ub = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    activation_coeffs = -common.conv_transpose(mu_out, w,
                                               lb.shape[1:].as_list(),
                                               padding, strides)
    dual_obj_bias = -tf.reduce_sum(mu_out * b, axis=(2, 3, 4))
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    # Compare against equivalent linear layer.
    dual_obj_lin = _materialised_conv_layer_dual_objective(
        w, b, padding, strides, lam_in, mu_out, lb, ub)

    with self.test_session() as session:
      dual_obj_val, dual_obj_lin_val = session.run((dual_obj, dual_obj_lin))
      self.assertAllClose(dual_obj_val, dual_obj_lin_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_conv1d_layer_dual_objective_shape(self, dtype):
    num_classes = 6
    batch_size = 23
    input_length = 13
    kernel_length = 3
    input_channels = 3
    output_channels = 5
    padding = 'VALID'
    strides = (2,)

    # Output dimensions, based on convolution settings.
    output_length = 6

    w = tf.placeholder(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.placeholder(dtype=dtype, shape=(output_channels,))
    lam_in = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, input_length, input_channels))
    mu_out = tf.placeholder(dtype=dtype, shape=(
        num_classes, batch_size, output_length, output_channels))
    lb = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    ub = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))

    activation_coeffs = -common.conv_transpose(mu_out, w,
                                               lb.shape[1:].as_list(),
                                               padding, strides)
    dual_obj_bias = -tf.reduce_sum(mu_out * b, axis=(2, 3))
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    self.assertEqual(dtype, dual_obj.dtype)
    self.assertEqual((num_classes, batch_size), dual_obj.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_conv1d_layer_dual_objective(self, dtype, tol):
    num_classes = 5
    batch_size = 53
    input_length = 13
    kernel_length = 5
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2,)

    # Output dimensions, based on convolution settings.
    output_length = 5

    w = tf.random_normal(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.random_normal(dtype=dtype, shape=(output_channels,))
    lam_in = tf.random_normal(dtype=dtype, shape=(
        num_classes, batch_size, input_length, input_channels))
    mu_out = tf.random_normal(dtype=dtype, shape=(
        num_classes, batch_size, output_length, output_channels))
    lb = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    ub = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    activation_coeffs = -common.conv_transpose(mu_out, w,
                                               lb.shape[1:].as_list(),
                                               padding, strides)
    dual_obj_bias = -tf.reduce_sum(mu_out * b, axis=(2, 3))
    dual_obj = standard_layer_calcs.linear_dual_objective(
        lam_in, activation_coeffs, dual_obj_bias, lb, ub)

    # Compare against equivalent linear layer.
    dual_obj_lin = _materialised_conv_layer_dual_objective(
        w, b, padding, strides, lam_in, mu_out, lb, ub)

    with self.test_session() as session:
      dual_obj_val, dual_obj_lin_val = session.run((dual_obj, dual_obj_lin))
      self.assertAllClose(dual_obj_val, dual_obj_lin_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      ('float32_snt', snt.BatchNorm, tf.float32, 1.e-5, False),
      ('float64_snt', snt.BatchNorm, tf.float64, 1.e-8, False),
      ('float32', ibp.BatchNorm, tf.float32, 1.e-5, False),
      ('float64', ibp.BatchNorm, tf.float64, 1.e-8, False),
      ('float32_train', ibp.BatchNorm, tf.float32, 1.e-5, True),
      ('float64_train', ibp.BatchNorm, tf.float64, 1.e-8, True))
  def test_batchnorm_layer_dual_objective(self, batchnorm_class, dtype, tol,
                                          is_training):
    num_classes = 3
    batch_size = 53
    input_size = 7
    output_size = 5

    mu_out = tf.random_normal(dtype=dtype, shape=(
        num_classes, batch_size, output_size))
    lb = tf.random_normal(dtype=dtype, shape=(batch_size, input_size))
    ub = tf.random_normal(dtype=dtype, shape=(batch_size, input_size))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    # Linear layer.
    w = tf.random_normal(dtype=dtype, shape=(input_size, output_size))
    b = tf.random_normal(dtype=dtype, shape=(output_size,))

    # Batch norm layer.
    epsilon = 1.e-2
    bn_initializers = {
        snt.BatchNorm.BETA: tf.random_normal_initializer(),
        snt.BatchNorm.GAMMA: tf.random_uniform_initializer(.1, 3.),
        snt.BatchNorm.MOVING_MEAN: tf.random_normal_initializer(),
        snt.BatchNorm.MOVING_VARIANCE: tf.random_uniform_initializer(.1, 3.)
    }
    batchnorm_module = batchnorm_class(offset=True, scale=True, eps=epsilon,
                                       initializers=bn_initializers)
    # Need to connect the batchnorm module to the graph.
    batchnorm_module(tf.random_normal(dtype=dtype,
                                      shape=(batch_size, output_size)),
                     is_training=is_training)

    # Calculate dual objective contribution of linear layer with batch norm.
    dual_obj, mu_bn = standard_layer_calcs.batchnorm_layer_dual_objective(
        batchnorm_module, mu_out)
    activation_coeffs = -tf.tensordot(mu_bn, tf.transpose(w), axes=1)
    dual_obj_bias = -tf.tensordot(mu_bn, b, axes=1)
    dual_obj += standard_layer_calcs.linear_dual_objective(
        None, activation_coeffs, dual_obj_bias, lb, ub)

    # Separately, calculate dual objective by adjusting the linear layer.
    wn, bn = common.combine_with_batchnorm(w, b, batchnorm_module)
    activation_coeffs_lin = -tf.tensordot(mu_out, tf.transpose(wn), axes=1)
    dual_obj_bias_lin = -tf.tensordot(mu_out, bn, axes=1)
    dual_obj_lin = standard_layer_calcs.linear_dual_objective(
        None, activation_coeffs_lin, dual_obj_bias_lin, lb, ub)

    init_op = tf.global_variables_initializer()

    with self.test_session() as session:
      session.run(init_op)
      # Verify that both methods give the same result.
      dual_obj_val, dual_obj_lin_val = session.run((dual_obj, dual_obj_lin))
      self.assertAllClose(dual_obj_val, dual_obj_lin_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(('plain', False), ('relu', True))
  def test_global_maxpool_layer_dual_objective_shape(self, with_relu):
    num_classes = 11
    batch_size = 6
    input_height = 17
    input_width = 7
    layer_channels = 3

    mu_in = tf.placeholder(dtype=tf.float32, shape=(
        num_classes, batch_size, input_height, input_width, layer_channels))
    lam_out = tf.placeholder(dtype=tf.float32, shape=(
        num_classes, batch_size, 1, 1, layer_channels))
    lb = tf.placeholder(dtype=tf.float32, shape=(
        batch_size, input_height, input_width, layer_channels))
    ub = tf.placeholder(dtype=tf.float32, shape=(
        batch_size, input_height, input_width, layer_channels))

    dual_obj = standard_layer_calcs.maxpool_layer_dual_objective(
        None, None, with_relu, mu_in, lam_out, lb, ub)

    self.assertEqual((num_classes, batch_size), dual_obj.shape)

  @parameterized.named_parameters(('plain', False), ('relu', True))
  def test_maxpool_layer_dual_objective_shape(self, with_relu):
    num_classes = 6
    batch_size = 7
    input_height = 33
    input_width = 20
    layer_channels = 3

    # Output dimensions, based on maxpool settings.
    output_height = 11
    output_width = 5

    mu_in = tf.placeholder(dtype=tf.float32, shape=(
        num_classes, batch_size, input_height, input_width, layer_channels))
    lam_out = tf.placeholder(dtype=tf.float32, shape=(
        num_classes, batch_size, output_height, output_width, layer_channels))
    lb = tf.placeholder(dtype=tf.float32, shape=(
        batch_size, input_height, input_width, layer_channels))
    ub = tf.placeholder(dtype=tf.float32, shape=(
        batch_size, input_height, input_width, layer_channels))

    dual_obj = standard_layer_calcs.maxpool_layer_dual_objective(
        [3, 4], (3, 4), with_relu, mu_in, lam_out, lb, ub)

    self.assertEqual((num_classes, batch_size), dual_obj.shape)

  @parameterized.named_parameters(('plain', False), ('relu', True))
  def test_global_maxpool_layer_dual_objective(self, with_relu):
    num_classes = 11
    batch_size = 23
    input_height = 5
    input_width = 7
    layer_channels = 3

    mu_in = tf.random_normal(shape=(
        num_classes, batch_size, input_height, input_width, layer_channels))
    lam_out = tf.random_normal(shape=(
        num_classes, batch_size, 1, 1, layer_channels))
    lb = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    ub = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    dual_obj = standard_layer_calcs.maxpool_layer_dual_objective(
        None, None, with_relu, mu_in, lam_out, lb, ub)

    # Calculate the maxpool dual objective a different way.
    dual_obj_alt = self._max_layer_dual_objective(lb, ub, mu_in, lam_out,
                                                  with_relu)

    init_op = tf.global_variables_initializer()

    with self.test_session() as session:
      session.run(init_op)
      # Verify that both methods give the same result.
      dual_obj_val, dual_obj_alt_val = session.run((dual_obj, dual_obj_alt))
      tol = 1.e-6
      self.assertAllClose(dual_obj_val, dual_obj_alt_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(('plain', False), ('relu', True))
  def test_maxpool_layer_dual_objective(self, with_relu):
    num_classes = 7
    batch_size = 13
    input_height = 14
    input_width = 15
    layer_channels = 3
    kernel_height = 2
    kernel_width = 3
    stride_vertical = 2
    stride_horizontal = 3

    # Output dimensions, based on maxpool settings.
    # This maxpool tiles perfectly.
    output_height = 7
    output_width = 5

    mu_in = tf.random_normal(shape=(
        num_classes, batch_size, input_height, input_width, layer_channels))
    lam_out = tf.random_normal(shape=(
        num_classes, batch_size, output_height, output_width, layer_channels))
    lb = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    ub = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    dual_obj = standard_layer_calcs.maxpool_layer_dual_objective(
        [kernel_height, kernel_width], (stride_vertical, stride_horizontal),
        with_relu, mu_in, lam_out, lb, ub)

    # Calculate the maxpool dual objective a different way.
    dual_obj_alt = 0.
    # Loop over all kernel placements.
    for output_row in range(output_height):
      for output_col in range(output_width):
        # Slice up the input tensors.
        output_row_slice = slice(output_row, output_row + 1)
        output_col_slice = slice(output_col, output_col + 1)
        input_row = stride_vertical * output_row
        input_row_slice = slice(input_row, input_row + kernel_height)
        input_col = stride_horizontal * output_col
        input_col_slice = slice(input_col, input_col + kernel_width)

        # Calculate contribution for this kernel placement.
        dual_obj_alt += self._max_layer_dual_objective(
            lb[:, input_row_slice, input_col_slice, :],
            ub[:, input_row_slice, input_col_slice, :],
            mu_in[:, :, input_row_slice, input_col_slice, :],
            lam_out[:, :, output_row_slice, output_col_slice, :],
            with_relu)

    init_op = tf.global_variables_initializer()

    with self.test_session() as session:
      session.run(init_op)
      # Verify that both methods give the same result.
      dual_obj_val, dual_obj_alt_val = session.run((dual_obj, dual_obj_alt))
      tol = 1.e-6
      self.assertAllClose(dual_obj_val, dual_obj_alt_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(('plain', False), ('relu', True))
  def test_overlapping_maxpool_layer_dual_objective(self, with_relu):
    num_classes = 11
    batch_size = 13
    input_height = 7
    input_width = 14
    layer_channels = 3
    kernel_height = 3
    kernel_width = 5
    stride_vertical = 1
    stride_horizontal = 3

    # Output dimensions, based on maxpool settings.
    # This maxpool has overlaps vertically and horizontally.
    output_height = 5
    output_width = 4
    vertical_overlap = tf.reshape(
        tf.constant([1, 2, 3, 3, 3, 2, 1],
                    dtype=tf.float32),
        shape=(input_height, 1, 1))
    horizontal_overlap = tf.reshape(
        tf.constant([1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1],
                    dtype=tf.float32),
        shape=(1, input_width, 1))

    mu_in = tf.random_normal(shape=(
        num_classes, batch_size, input_height, input_width, layer_channels))
    lam_out = tf.random_normal(shape=(
        num_classes, batch_size, output_height, output_width, layer_channels))
    lb = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    ub = tf.random_normal(shape=(
        batch_size, input_height, input_width, layer_channels))
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    dual_obj = standard_layer_calcs.maxpool_layer_dual_objective(
        [kernel_height, kernel_width], (stride_vertical, stride_horizontal),
        with_relu, mu_in, lam_out, lb, ub)

    # Calculate the maxpool dual objective a different way.
    # Share the inputs' dual variables equally amongst all pools they belong to.
    mu_in_shared = mu_in / (vertical_overlap * horizontal_overlap)
    dual_obj_alt = 0.
    # Loop over all kernel placements.
    for output_row in range(output_height):
      for output_col in range(output_width):
        # Slice up the input tensors.
        output_row_slice = slice(output_row, output_row + 1)
        output_col_slice = slice(output_col, output_col + 1)
        input_row = stride_vertical * output_row
        input_row_slice = slice(input_row, input_row + kernel_height)
        input_col = stride_horizontal * output_col
        input_col_slice = slice(input_col, input_col + kernel_width)

        # Calculate contribution for this kernel placement.
        dual_obj_alt += self._max_layer_dual_objective(
            lb[:, input_row_slice, input_col_slice, :],
            ub[:, input_row_slice, input_col_slice, :],
            mu_in_shared[:, :, input_row_slice, input_col_slice, :],
            lam_out[:, :, output_row_slice, output_col_slice, :],
            with_relu)

    init_op = tf.global_variables_initializer()

    with self.test_session() as session:
      session.run(init_op)
      # Verify that both methods give the same result.
      dual_obj_val, dual_obj_alt_val = session.run((dual_obj, dual_obj_alt))
      tol = 1.e-6
      self.assertAllClose(dual_obj_val, dual_obj_alt_val, atol=tol, rtol=tol)

  def _max_layer_dual_objective(self, lb, ub, mu_in, lam_out, with_relu):
    """Calculates expected dual objective for a global 'max' layer.

    This can also be called repeatedly to obtain the dual objective for a
    maxpool with a moving kernel that tiles the input space without overlapping.

    Maximises (over y in [lb, ub])::
      mu^T y  -  lam max(y)
    by conditioning on the max obtaining its maximum at y_i, and evaluating
    separately at the vertices of the resulting piecewise-linear function
    in y_i.

    Args:
      lb: 4D tensor of shape (batch_size, height, width, channels)
        containing lower bounds on the inputs.
      ub: 4D tensor of shape (batch_size, height, width, channels)
        containing upper bounds on the inputs.
      mu_in: 5D tensor of shape (num_classes, batch_size, height, width,
        channels) dual variables for the inputs' preceding linear calculations.
      lam_out: 5D tensor of shape (num_classes, batch_size, 1, 1, channels)
        containing dual variables for the outputs' max-pool calculations.
      with_relu: Boolean, whether to apply a relu to the maxpool.

    Returns:
      2D tensor of shape (num_classes, batch_size) containing the dual
        objective for the 'max' layer for each example.
    """
    # Recall the problem: maximise (over y in [lb, ub])::
    #   mu^T y  -  lam max(y)
    #
    # For each input (kernel element) i, we condition on the maxpool
    # attaining its maximum at y_i.
    # This leads us to maximise (over z_j in [lb_j, min{y_i, ub_j}]
    # and constraining z_i=y_i)::
    #   mu^T z  -  lam y_i
    #
    # That maximum, as a function of y_i in the domain [lb_max, ub_i],
    # is concave and piecewise linear with cusps at ub_k.
    # Instead of bisection, we evaluate it at those values of ub_k that lie
    # within the interval [lb_max, ub_i], and also at lb_max itself.
    # The maximum over all such k and i is our dual objective.

    lb_max = tf.reduce_max(lb, axis=[1, 2], keepdims=True)
    if with_relu:
      # Model ReLU as an additional fixed input to the max, with bounds [0, 0].
      lb_max = tf.maximum(lb_max, 0.)

    # We'll need to consider ub_i and ub_k together.
    # Set up xx_i, xx_k tensors shaped as (class?, N, Hi, Wi, Hk, Wk, C)
    # where Hi, Hk range over input rows and Wi, Wk range over input columns.
    mu_i = tf.expand_dims(tf.expand_dims(mu_in, 4), 5)
    lam = tf.expand_dims(tf.expand_dims(lam_out, 4), 5)
    lb_i = tf.expand_dims(tf.expand_dims(lb, 3), 4)
    ub_i = tf.expand_dims(tf.expand_dims(ub, 3), 4)
    ub_k = tf.expand_dims(tf.expand_dims(ub, 1), 2)
    lb_max = tf.expand_dims(tf.expand_dims(lb_max, 1), 2)

    def dual_obj_given_max_at_yi(c_k):
      # Evaluate max (mu^T z - lam y_i) at y_i = c_k.
      bc = tf.zeros_like(mu_i + c_k)  # tf.where doesn't broadcast
      dual_obj = standard_layer_calcs.max_linear(
          mu_i + bc, lb_i + bc,
          tf.minimum(c_k, ub_i), axis=[2, 3], keepdims=True)
      dual_obj -= tf.maximum(lb_i * mu_i, tf.minimum(c_k, ub_i) * mu_i)
      dual_obj += (mu_i - lam) * c_k
      # Only consider this y if it's in the permitted range for a maximal y_i.
      feasible = tf.logical_and(lb_max <= c_k, c_k <= ub_i + bc)
      dual_obj = tf.where(feasible, dual_obj, -1.e8 + bc)
      # Take maximum over all i, k.
      return tf.reduce_max(dual_obj, axis=[2, 3, 4, 5])

    # Evaluate max (mu^T z - lam y_i) at y_i = max_j lb_j.
    dual_obj_lb_max = dual_obj_given_max_at_yi(lb_max)
    # Evaluate max (mu^T z - lam y_i) at y_i = ub_k.
    dual_obj_ub_k = dual_obj_given_max_at_yi(ub_k)

    dual_obj_max = tf.maximum(dual_obj_lb_max, dual_obj_ub_k)

    if with_relu:
      # Also consider the case in which all y_i are <= 0,
      # so relu_maxpool is zero.
      # This leads us to maximise (over z_j in [lb_j, min{0, ub_j}])::
      #   mu^T z  -  lam 0

      bc = tf.zeros_like(mu_i)  # tf.where doesn't broadcast
      dual_obj_zero = standard_layer_calcs.max_linear(
          mu_i + bc, lb_i + bc,
          tf.minimum(0., ub_i), axis=[2, 3], keepdims=True)
      # Only consider this case if the y_i can actually all be <= 0.
      bc = tf.zeros_like(dual_obj_zero)  # tf.where doesn't broadcast
      feasible = lb_max <= bc
      dual_obj_zero = tf.where(feasible, dual_obj_zero, -1.e8 + bc)
      dual_obj_zero = tf.reduce_max(dual_obj_zero, axis=[2, 3, 4, 5])
      dual_obj_max = tf.maximum(dual_obj_max, dual_obj_zero)

    return tf.reduce_sum(dual_obj_max, axis=2)


def _materialised_conv_layer_dual_objective(w, b, padding, strides,
                                            lam_in, mu_out, lb, ub):
  """Materialised version of `conv_layer_dual_objective`."""
  # Flatten the inputs, as the materialised convolution will have no
  # spatial structure.
  mu_out_flat = snt.BatchFlatten(preserve_dims=2)(mu_out)

  # Materialise the convolution as a (sparse) fully connected linear layer.
  w_flat, b_flat = common.materialise_conv(w, b, lb.shape[1:].as_list(),
                                           padding=padding, strides=strides)

  activation_coeffs = -tf.tensordot(mu_out_flat, tf.transpose(w_flat), axes=1)
  dual_obj_bias = -tf.tensordot(mu_out_flat, b_flat, axes=1)

  # Flatten the inputs, as the materialised convolution will have no
  # spatial structure.
  if lam_in is not None:
    lam_in = snt.FlattenTrailingDimensions(2)(lam_in)
  lb = snt.BatchFlatten()(lb)
  ub = snt.BatchFlatten()(ub)

  return standard_layer_calcs.linear_dual_objective(
      lam_in, activation_coeffs, dual_obj_bias, lb, ub)


if __name__ == '__main__':
  tf.test.main()
