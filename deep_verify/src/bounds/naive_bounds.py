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

"""Naive bound calculation for common neural network layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.bounds import layer_bounds
import interval_bound_propagation as ibp
import tensorflow as tf


def input_bounds(inputs, delta, lower_bound=0., upper_bound=1.,
                 preprocess_fn=None):
  """Calculates interval bounds on the network inputs.

  Args:
    inputs: 2D tensor of shape (batch_size, input_size), or 4D tensor of
      shape (batch_size, height, width, channels), of input examples.
    delta: Permitted perturbation on each input.
    lower_bound: Scalar - smallest permissible input (pixel) value.
    upper_bound: Scalar - largest permissible input (pixel) value.
    preprocess_fn: Optional function mapping tensor to tensor
      performing pre-processing on the raw inputs.

  Returns:
    `IntervalBounds` for the inputs, relative to `inputs`.
  """
  # Input range, according to permitted perturbation radius.
  if preprocess_fn:
    lb = preprocess_fn(tf.maximum(inputs - delta, lower_bound)) - inputs
    ub = preprocess_fn(tf.minimum(inputs + delta, upper_bound)) - inputs
  else:
    lb = tf.maximum(-delta, lower_bound - inputs)
    ub = tf.minimum(delta, upper_bound - inputs)
  return ibp.RelativeIntervalBounds(lb, ub, inputs)


class NaiveBoundPropagation(layer_bounds.BoundPropagation):
  """Naive layer-wise bound propagation method."""
