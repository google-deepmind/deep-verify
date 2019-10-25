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

"""Calculations relating to the Gram matrix of a convolution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src import common
from interval_bound_propagation import layer_utils
import numpy as np
import tensorflow as tf


def conv_weighted_gram_matrix(w, d, input_shape, padding, strides,
                              w_s=None):
  """Calculates W^T d W for an N-D convolution W, exploiting sparsity.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width,
      input_channels, output_channels) containing the convolutional kernel.
    d: (N+3)D tensor of shape (num_targets, batch_size,
      output_height, output_width, output_channels), interpreted as a
      diagonal weight matrix.
    input_shape: List of length N+1 specifying
      [input_height, input_width, input_channels].
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    w_s: Optional (N+2)D tensor of shape (kernel_height, kernel_width,
      input_slice_channels, output_channels) containing a slice of `w`
      (over input_channels) if it is desired to build the Gram matrix a few
      columns at a time; defaults to `w` to build the Gram matrix in full.

  Returns:
    (2N+4)D tensor of shape (num_targets, batch_size,
    input_height, input_width, input_slice_channels,
    2*kernel_height-1, 2*kernel_width-1, input_channels)
    expressing W^T d W in a sheared form to exploit sparsity.
  """
  w_s = w_s if w_s is not None else w
  num_targets = d.shape[0].value
  batch_size = tf.shape(d)[1]
  n = w.shape.ndims - 2
  kernel_shape = w.shape[:-2].as_list()
  input_channels = w.shape[-2].value
  input_slice_channels = w_s.shape[-2].value
  output_channels = w.shape[-1].value
  enlarged_kernel_shape = [2*s-1 for s in kernel_shape]

  # We wish to combine W with itself at different kernel offsets,
  # from -kernel_size to +kernel_size (exclusive).
  # Achieve this by considering W (kernel) as a new stride-1 deconvolution.
  w_offset, _ = layer_utils.materialise_conv(
      tf.reverse(w, axis=list(range(n))), None,
      input_shape=(enlarged_kernel_shape + [-1]),
      padding='VALID', strides=(n*[1]))
  # The above materialises it as a 2D tensor with shape
  # (enlarged_kernel_shape*input_channels,
  # kernel_height*kernel_width*output_channels).
  w_offset = tf.reshape(w_offset, shape=(
      [1] + enlarged_kernel_shape + [input_channels] +
      kernel_shape + [output_channels]))
  w_offset = tf.transpose(tf.reverse(w_offset, axis=list(range(1, n+1))),
                          perm=(list(range(n+2, 2*n+2)) +
                                list(range(n+2)) + [2*n+2]))
  # W_offset is now a (2N+3)D tensor with shape
  # (kernel_height, kernel_width, 1,
  # 2*kernel_height-1, 2*kernel_width-1, input_channels, output_channels).

  # Take all relevant pair-wise products of W with W_offset.
  wtw = w_offset * tf.reshape(w_s, shape=(
      kernel_shape + [input_slice_channels] + (n*[1]) + [1, output_channels]))
  # WTW is a (2N+3)D tensor with shape
  # (kernel_height, kernel_width, input_slice_channels,
  # 2*kernel_height-1, 2*kernel_width-1, input_channels, output_channels).

  # Combine with d, by performing a deconvolution.
  wtw = tf.reshape(wtw, shape=(
      kernel_shape +
      [input_slice_channels*np.prod(enlarged_kernel_shape)*input_channels,
       output_channels]))
  result = common.conv_transpose(d, wtw,
                                 input_shape[:-1] + [wtw.shape[n].value],
                                 padding, strides)
  # Output from common.conv_transpose is of shape:
  # (num_targets, batch_size, input_height, input_width,
  # input_slice_channels*enlarged_kernel_shape*input_channels).
  result = tf.reshape(result, shape=(
      [num_targets, batch_size] + input_shape[:-1] + [input_slice_channels] +
      enlarged_kernel_shape + [input_channels]))
  # Return a (2N+4)D tensor of shape (num_targets, batch_size,
  # input_height, input_width, input_slice_channels,
  # 2*kernel_height-1, 2*kernel_width-1, input_channels).
  return result


def _conv_projection_slice(w, d, w_s, beta, padding, strides,
                           abs_fn=tf.abs):
  """Calculates a partial projection of | W^T d W |.

  Computes  sum_j |Q_ij| beta_j  where  Q = W^T d W_s  is a slice of the
  weighted Gram matrix and j indexes the column channels included in the slice.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width,
      input_channels, output_channels) containing the convolutional kernel.
    d: (N+3)D tensor of shape (num_targets, batch_size,
      output_height, output_width, output_channels), interpreted as a
      diagonal weight matrix.
    w_s: (N+2)D tensor of shape (kernel_height, kernel_width,
      input_slice_channels, output_channels) containing the
      desired slice of `w`.
    beta: (N+3)D tensor of shape (num_targets, batch_size, input_height,
      input_width, input_slice_channels) specifying the projection.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    abs_fn: Absolute value function; defaults to `tf.abs`.

  Returns:
    (N+3)D tensor of shape (num_targets, batch_size,
    input_height, input_width, input_slice_channels) containing
    sum_j |Q_ij| beta_j.
  """
  num_targets = d.shape[0].value
  batch_size = tf.shape(d)[1]
  n = w.shape.ndims - 2
  input_shape = beta.shape[2:].as_list()

  wt_d_w = conv_weighted_gram_matrix(w, d, input_shape, padding, strides,
                                     w_s=w_s)
  # wt_d_w is a (2N+4)D tensor of shape (num_targets, batch_size,
  # input_height, input_width, input_slice_channels,
  # 2*kernel_height-1, 2*kernel_width-1, input_channels).

  a = abs_fn(wt_d_w) * tf.reshape(beta, shape=(
      [num_targets, batch_size] + input_shape + (n*[1]) + [1]))
  return common.conv_reduce_sum(a, input_shape,
                                padding='SAME', strides=(n*[1]))


def conv_weighted_gram_abs_projection(w, d, beta, padding, strides,
                                      abs_fn=tf.abs,
                                      block_size=0):
  """Calculates a projection of | W^T d W | for an N-D convolution W.

  Computes  beta_i^{-1} sum_j |Q_ij| beta_j  where  Q = W^T d W  is the
  weighted Gram matrix for the convolution.

  The computation exploits sparsity of the convolution, thereby managing
  to run in  O(K^2 M C^3 + K^3 M C^2)  time per example, for C channels,
  spatial size M, and kernel size K. By comparison, working with
  a fully materialised MCxMC matrix would require  O(M^3 C^3) time.

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
    abs_fn: Absolute value function; defaults to `tf.abs`.
    block_size: Number of column channels of  W^T d W  to process at a time,
      or zero (default) to process all at once.

  Returns:
    (N+3)D tensor of shape (num_targets, batch_size,
    input_height, input_width, input_channels) containing
    beta_i^{-1} sum_j |Q_ij| beta_j.
  """
  if block_size == 0:
    proj = _conv_projection_slice(w, d, w, beta,
                                  padding, strides, abs_fn=abs_fn)

  else:
    # Accumulate over slices of the input channels dimension.
    input_channels = w.shape[-2].value
    proj = tf.zeros_like(beta)
    for idx_min in range(0, input_channels, block_size):
      idx_max = min(idx_min+block_size, input_channels)
      if beta.shape.ndims == 5:
        w_s = w[:, :, idx_min:idx_max]
        beta_s = beta[:, :, :, :, idx_min:idx_max]
      elif beta.shape.ndims == 4:
        w_s = w[:, idx_min:idx_max]
        beta_s = beta[:, :, :, idx_min:idx_max]
      else:
        raise ValueError('Invalid rank for beta: {}'.format(beta.shape.ndims))
      proj += _conv_projection_slice(w, d, w_s, beta_s,
                                     padding, strides, abs_fn=abs_fn)

  return proj / beta


def _linear_projection_slice(w, d, w_s, beta, abs_fn=tf.abs):
  """Calculates a partial projection of | W^T d W |.

  Computes  sum_j |Q_ij| beta_j  where  Q = W^T d W_s  is a slice of the
  weighted Gram matrix and j indexes the columns included in the slice.

  Args:
    w: 2D tensor of shape (input_size, output_size) containing the matrix.
    d: 3D tensor of shape (num_targets, batch_size, output_size),
      interpreted as a diagonal weight matrix.
    w_s: 2D tensor of shape (input_slice_size, output_size) containing the
      desired slice of `w`.
    beta: 3D tensor of shape (num_targets, batch_size, input_slice_size)
      specifying the partial projection.
    abs_fn: Absolute value function; defaults to `tf.abs`.

  Returns:
    3D tensor of shape (num_targets, batch_size, input_size)
    containing sum_j |Q_ij| beta_j.
  """
  dw = tf.expand_dims(d, axis=2) * w_s
  wt_d_w = tf.matmul(w, dw, transpose_b=True)
  # wt_d_w is a 4D tensor of shape (num_targets, batch_size,
  # input_size, input_slice_size).

  return tf.reduce_sum(abs_fn(wt_d_w) * tf.expand_dims(beta, axis=2), axis=3)


def linear_weighted_gram_abs_projection(w, d, beta,
                                        abs_fn=tf.abs,
                                        block_size=0):
  """Calculates a projection of | W^T d W | for a fully connected matrix W.

  Computes  beta_i^{-1} sum_j |Q_ij| beta_j  where  Q = W^T d W  is the
  weighted Gram matrix.

  Args:
    w: 2D tensor of shape (input_size, output_size) containing the matrix.
    d: 3D tensor of shape (num_targets, batch_size, output_size),
      interpreted as a diagonal weight matrix.
    beta: 3D tensor of shape (num_targets, batch_size, input_size)
      specifying the projection.
    abs_fn: Absolute value function; defaults to `tf.abs`.
    block_size: Number of columns of  W^T d W  to process at a time,
      or zero (default) to process all at once.

  Returns:
    3D tensor of shape (num_targets, batch_size, input_size)
    containing beta_i^{-1} sum_j |Q_ij| beta_j.
  """
  # Normalise beta.
  beta = beta / tf.reduce_sum(beta, axis=2, keepdims=True)

  if block_size == 0:
    proj = _linear_projection_slice(w, d, w, beta, abs_fn=abs_fn)

  else:
    # Accumulate over slices of the input dimension.
    input_size = w.shape[0].value
    proj = tf.zeros_like(beta)
    for idx_min in range(0, input_size, block_size):
      idx_max = min(idx_min+block_size, input_size)
      w_s = w[idx_min:idx_max]
      beta_s = beta[:, :, idx_min:idx_max]
      proj += _linear_projection_slice(w, d, w_s, beta_s,
                                       abs_fn=abs_fn)

  return proj / beta


