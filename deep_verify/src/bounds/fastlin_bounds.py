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

"""Fastlin bound calculation for common neural network layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.bounds import layer_bounds
import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


class SymbolicBounds(ibp.SymbolicBounds):
  """Upper and lower bounds, as linear expressions in the original inputs."""

  def __init__(self, lower, upper, nominal):
    super(SymbolicBounds, self).__init__(lower, upper)
    self._nominal = nominal

  @staticmethod
  def convert(bounds):
    if isinstance(bounds, SymbolicBounds):
      return bounds

    if isinstance(bounds, tf.Tensor):
      nominal = bounds
    else:
      nominal = bounds.nominal
      bounds = ibp.IntervalBounds(bounds.lower, bounds.upper)

    symbolic_bounds = ibp.SymbolicBounds.convert(bounds)
    return SymbolicBounds(symbolic_bounds.lower, symbolic_bounds.upper,
                          nominal)

  def apply_batch_reshape(self, wrapper, shape):
    bounds_out = super(SymbolicBounds, self).apply_batch_reshape(wrapper, shape)
    nominal_out = snt.BatchReshape(shape)(self._nominal)
    return SymbolicBounds(bounds_out.lower, bounds_out.upper,
                          nominal_out).with_priors(wrapper.output_bounds)

  def apply_linear(self, wrapper, w, b):
    bounds_out = super(SymbolicBounds, self).apply_linear(wrapper, w, b)

    nominal_out = tf.matmul(self._nominal, w)
    if b is not None:
      nominal_out += b

    return SymbolicBounds(bounds_out.lower, bounds_out.upper,
                          nominal_out).with_priors(wrapper.output_bounds)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    bounds_out = super(SymbolicBounds, self).apply_conv2d(wrapper, w, b,
                                                          padding, strides)

    nominal_out = tf.nn.convolution(self._nominal, w,
                                    padding=padding, strides=strides)
    if b is not None:
      nominal_out += b

    return SymbolicBounds(bounds_out.lower, bounds_out.upper,
                          nominal_out).with_priors(wrapper.output_bounds)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    bounds_out = super(SymbolicBounds, self).apply_increasing_monotonic_fn(
        wrapper, fn, *args, **parameters)
    nominal_out = fn(self._nominal)
    return SymbolicBounds(bounds_out.lower, bounds_out.upper,
                          nominal_out).with_priors(wrapper.output_bounds)

  def concretize(self):
    """Concretize activation bounds."""
    if self._concretized is None:
      concrete = super(SymbolicBounds, self).concretize()
      self._concretized = ConcretizedBounds(concrete.lower, concrete.upper,
                                            self._nominal)
    return self._concretized


class ConcretizedBounds(ibp.IntervalBounds):
  """Concretised interval bounds with nominals."""

  def __init__(self, lower, upper, nominal):
    super(ConcretizedBounds, self).__init__(lower, upper)
    self._nominal = nominal
    self._update_cached_bounds_op = None

  @property
  def nominal(self):
    return self._nominal

  @property
  def lower_rel(self):
    """Returns lower bounds, expressed relative to nominal values."""
    return self.lower - self.nominal

  @property
  def upper_rel(self):
    """Returns upper bounds, expressed relative to nominal values."""
    return self.upper - self.nominal

  def _set_up_cache(self):
    self._lower, update_lower_op = self._cache_with_update_op(self._lower)
    self._upper, update_upper_op = self._cache_with_update_op(self._upper)
    return tf.group([update_lower_op, update_upper_op])


class FastlinBoundPropagation(layer_bounds.BoundPropagation):
  """Method for propagating symbolic bounds in multiple passes."""

  def __init__(self, num_rounds=1, best_with_naive=False):
    super(FastlinBoundPropagation, self).__init__()
    self._num_rounds = num_rounds
    self._best_with_naive = best_with_naive

  def propagate_bounds(self, network, in_bounds):
    if self._best_with_naive:
      # Initial round of interval bound propagation.
      super(FastlinBoundPropagation, self).propagate_bounds(network, in_bounds)

    for _ in range(self._num_rounds):
      # Construct symbolic bounds and propagate them.
      super(FastlinBoundPropagation, self).propagate_bounds(
          network, SymbolicBounds.convert(in_bounds))
