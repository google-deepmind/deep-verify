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

"""Tests for Gram matrix computations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deep_verify.src import common
from deep_verify.src.formulations.semidefinite import gram_calcs
import sonnet as snt
import tensorflow as tf


def conv_weighted_gram_abs_projection_slow(w, d, beta, padding, strides):
  """Calculates a projection of | W^T d W | for an N-D convolution W.

  Computes  beta_i^{-1} sum_j |Q_ij| beta_j  where  Q = W^T d W  is the
  weighted Gram matrix for the convolution.

  The convolution exploits sparsity of the convolution, thereby managing
  to run in  O(K^2 M C^3 + K^3 M C^2)  time per example, for C channels,
  spatial size M, and kernel size K. By comparison, working with
  a fully materialised MCxMC matrix would require  O(M^3 C^3)  time.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width,
      input_channels, output_channels) containing the convolutional kernel.
    d: (N+3)D tensor of shape (num_targets, batch_size,
      output_height, output_width, output_channels), interpreted as a
      diagonal weight matrix.
    beta: (N+3)D tensor of shape (num_targets, batch_size,
      input_height, input_width, input_channels) specifying the projection.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    (N+3)D tensor of shape (num_targets, batch_size,
    input_height, input_width, input_channels) containing | W^T d W | beta.
  """
  input_shape = beta.shape[2:].as_list()

  flatten = snt.BatchFlatten(preserve_dims=2)
  unflatten = snt.BatchReshape(input_shape, preserve_dims=2)

  w_lin, _ = common.materialise_conv(w, None, input_shape,
                                     padding, strides)
  return unflatten(linear_weighted_gram_abs_projection_slow(w_lin,
                                                            flatten(d),
                                                            flatten(beta)))


def linear_weighted_gram_abs_projection_slow(w, d, beta):
  """Calculates a projection of | W^T d W | for a fully connected matrix W.

  Computes  beta_i^{-1} sum_j |Q_ij| beta_j  where  Q = W^T d W  is the
  weighted Gram matrix.

  Args:
    w: 2D tensor of shape (input_size, output_size) containing the matrix.
    d: 3D tensor of shape (num_targets, batch_size, output_size),
      interpreted as a diagonal weight matrix.
    beta: 3D tensor of shape (num_targets, batch_size, input_size)
      specifying the projection.

  Returns:
    3D tensor of shape (num_targets, batch_size, input_size)
    containing | W^T d W | beta.
  """
  q = tf.einsum('cnio,cnjo->cnij',
                w * tf.expand_dims(d, axis=2),
                w * tf.expand_dims(beta, axis=3))
  return tf.reduce_sum(tf.abs(q), axis=3) / beta


class GramCalcsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('SAME_32', 11, 3, 11, 'SAME', [1], tf.float32, 1.e-5),
      ('SAME', 17, 3, 17, 'SAME', [1]),
      ('SAME_even', 17, 2, 17, 'SAME', [1]),
      ('SAME_strided', 17, 4, 9, 'SAME', [2]),
      ('SAME_blocked', 11, 3, 11, 'SAME', [1], tf.float32, 1.e-5, 4),
      ('VALID_32', 11, 3, 9, 'VALID', [1], tf.float32, 1.e-5),
      ('VALID', 17, 3, 15, 'VALID', [1]),
      ('VALID_even', 17, 2, 16, 'VALID', [1]),
      ('VALID_strided', 17, 4, 7, 'VALID', [2]))
  def test_conv1d_weighted_gram(self, input_size, kernel_size, output_size,
                                padding, strides,
                                dtype=tf.float64, atol=1.e-9, block_size=0):
    num_targets = 3
    batch_size = 2
    input_channels = 13
    output_channels = 11

    w = tf.random_normal(dtype=dtype,
                         shape=[kernel_size,
                                input_channels, output_channels])
    d = tf.random_gamma(alpha=1, dtype=dtype,
                        shape=[num_targets, batch_size,
                               output_size, output_channels])
    beta = tf.random_gamma(alpha=1, dtype=dtype,
                           shape=[num_targets, batch_size,
                                  input_size, input_channels])

    proj = gram_calcs.conv_weighted_gram_abs_projection(w, d, beta,
                                                        padding, strides,
                                                        block_size=block_size)
    proj_slow = conv_weighted_gram_abs_projection_slow(w, d, beta,
                                                       padding, strides)

    with tf.Session() as session:
      proj_val, proj_slow_val = session.run((proj, proj_slow))
      self.assertAllClose(proj_val, proj_slow_val, atol=atol)

  @parameterized.named_parameters(
      ('SAME_32', (7, 11), (3, 3), (7, 11), 'SAME', [1, 1], tf.float32, 1.e-5),
      ('SAME', (7, 17), (3, 3), (7, 17), 'SAME', [1, 1]),
      ('SAME_even', (7, 17), (2, 2), (7, 17), 'SAME', [1, 1]),
      ('SAME_strided', (7, 17), (3, 4), (4, 9), 'SAME', [2, 2]),
      ('SAME_blocked', (7, 11), (3, 3), (7, 11), 'SAME', [1, 1],
       tf.float32, 1.e-5, 4),
      ('VALID_32', (7, 11), (3, 3), (5, 9), 'VALID', [1, 1], tf.float32, 1.e-5),
      ('VALID', (7, 17), (3, 3), (5, 15), 'VALID', [1, 1]),
      ('VALID_even', (7, 17), (2, 2), (6, 16), 'VALID', [1, 1]),
      ('VALID_strided', (7, 17), (3, 4), (3, 7), 'VALID', [2, 2]))
  def test_conv2d_weighted_gram(self, input_size, kernel_size, output_size,
                                padding, strides,
                                dtype=tf.float64, atol=1.e-9, block_size=0):
    num_targets = 3
    batch_size = 2
    input_height, input_width = input_size
    kernel_height, kernel_width = kernel_size
    input_channels = 13
    output_height, output_width = output_size
    output_channels = 11

    w = tf.random_normal(dtype=dtype,
                         shape=[kernel_height, kernel_width,
                                input_channels, output_channels])
    d = tf.random_gamma(alpha=1, dtype=dtype,
                        shape=[num_targets, batch_size,
                               output_height, output_width, output_channels])
    beta = tf.random_gamma(alpha=1, dtype=dtype,
                           shape=[num_targets, batch_size,
                                  input_height, input_width, input_channels])

    proj = gram_calcs.conv_weighted_gram_abs_projection(w, d, beta,
                                                        padding, strides,
                                                        block_size=block_size)
    proj_slow = conv_weighted_gram_abs_projection_slow(w, d, beta,
                                                       padding, strides)

    with tf.Session() as session:
      proj_val, proj_slow_val = session.run((proj, proj_slow))
      self.assertAllClose(proj_val, proj_slow_val, atol=atol)

  @parameterized.named_parameters(
      ('small_32', 7, 4, tf.float32, 1.e-5),
      ('small', 7, 4),
      ('large', 11, 13),
      ('large_blocked', 11, 13, tf.float32, 1.e-5, 3))
  def test_linear_weighted_gram(self, input_size, output_size,
                                dtype=tf.float64, atol=1.e-9, block_size=0):
    num_targets = 5
    batch_size = 3

    w = tf.random_normal(dtype=dtype, shape=[input_size, output_size])
    d = tf.random_gamma(alpha=1, dtype=dtype,
                        shape=[num_targets, batch_size, output_size])
    beta = tf.random_gamma(alpha=1, dtype=dtype,
                           shape=[num_targets, batch_size, input_size])

    proj = gram_calcs.linear_weighted_gram_abs_projection(w, d, beta,
                                                          block_size=block_size)
    proj_slow = linear_weighted_gram_abs_projection_slow(w, d, beta)

    with tf.Session() as session:
      proj_val, proj_slow_val = session.run((proj, proj_slow))
      self.assertAllClose(proj_val, proj_slow_val, atol=atol)


if __name__ == '__main__':
  tf.test.main()
