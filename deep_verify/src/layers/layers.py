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

"""Defines wrappers to easily propagate bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from deep_verify.src import common
from interval_bound_propagation import layer_utils
import six
import sonnet as snt
import tensorflow as tf


# Holds objective weights.
# The last layer can be combined with the target vector `c`.
ObjectiveWeights = collections.namedtuple('ObjectiveWeights', ['w', 'b'])


@six.add_metaclass(abc.ABCMeta)
class VerifiableLayer(object):
  """Abstract class for dual layers."""

  def __init__(self):
    self._no_duals = False

  @property
  def branches(self):
    """Returns list of (name, sub-layers list) pairs, e.g. for ResNet block."""
    return []

  @abc.abstractproperty
  def input_node(self):
    """Returns an `ibp.VerifiableWrapper` for the previous layer's outputs."""

  @abc.abstractproperty
  def output_node(self):
    """Returns an `ibp.VerifiableWrapper` for this layer's outputs."""

  @property
  def input_shape(self):
    return self.input_bounds.shape[1:]

  @property
  def output_shape(self):
    return self.output_bounds.shape[1:]

  @property
  def inputs(self):
    return self.input_bounds.nominal

  @property
  def outputs(self):
    return self.output_bounds.nominal

  @property
  def input_bounds(self):
    return self.input_node.output_bounds.concretize()

  @property
  def output_bounds(self):
    return self.output_node.output_bounds.concretize()

  def dual_shape(self):
    """Returns shape of the dual variable, or possibly nested dict thereof."""
    # By default, there is one dual variable for each output.
    return None if self._no_duals else tuple(self.output_shape)

  def set_no_duals(self):
    """Declares that this layer has no dual variables of its own."""
    self._no_duals = True

  def reshape_duals_forwards(self, next_layer, dual_vars):
    if next_layer.reshape:
      # There was a reshape prior to the next layer.
      dual_vars = snt.BatchReshape(next_layer.input_shape,
                                   preserve_dims=2)(dual_vars)
    return dual_vars

  def project_duals_op(self, dual_vars):  # pylint:disable=unused-argument
    """Projects duals into their regional constraints.

    Args:
      dual_vars: Dual variable tensor.

    Returns:
      Assignment op to modify `dual_vars`, clamping the dual variable
      values to their admissible ranges.
    """
    # By default, do no projection.
    return tf.no_op()


@six.add_metaclass(abc.ABCMeta)
class CustomOp(object):
  """Function or operation with a different implementation for each layer type.

  Each `visit_xxx` method is a call-back invoked via
  `SingleVerfiableLayer.custom_op`. They have implementation-specific *args and
  **kwargs, passed through by `SingleVerifiableLayer.custom_op`, for convenience
  so that the same visitor instance can be used multiple times with different
  arguments.
  """

  def visit_linear(self, layer, w, b, *args, **kwargs):
    """Callback for `Linear`."""
    raise NotImplementedError()

  def visit_conv(self, layer, w, b, padding, strides, *args, **kwargs):
    """Callback for `Conv`."""
    raise NotImplementedError()

  def visit_avgpool(self, layer, *args, **kwargs):
    """Callback for `AvgPool`."""
    raise NotImplementedError('AvgPool layers are not supported')

  def visit_maxpool(self, layer, *args, **kwargs):
    """Callback for `MaxPool`."""
    raise NotImplementedError('MaxPool layers are not supported')

  @abc.abstractmethod
  def visit_activation(self, layer, *args, **kwargs):
    """Callback for `Activation`."""


@six.add_metaclass(abc.ABCMeta)
class SingleVerifiableLayer(VerifiableLayer):
  """Dual layer for a single layer of the underlying network."""

  def __init__(self, input_node, output_node, module,
               batch_norm=None, reshape=False):
    super(SingleVerifiableLayer, self).__init__()
    self._module = module
    self._batch_norm = batch_norm
    self._reshape = reshape
    self._input_node = input_node
    self._output_node = output_node

  @property
  def input_node(self):
    return self._input_node

  @property
  def output_node(self):
    return self._output_node

  @abc.abstractproperty
  def is_activation(self):
    """Returns whether an activation layer, as opposed to linear/conv."""

  @property
  def module(self):
    return self._module

  @property
  def batch_norm(self):
    return self._batch_norm

  @property
  def reshape(self):
    return self._reshape

  def backward_prop_batchnorm(self, y):
    if self.batch_norm is not None:
      w, _ = layer_utils.decode_batchnorm(self.batch_norm)
      y = y * tf.cast(w, y.dtype)
    return y

  def backward_prop_batchnorm_bias(self, y, bias):
    if self.batch_norm is not None:
      w, b = layer_utils.decode_batchnorm(self.batch_norm)
      bias = bias + tf.reduce_sum(y * tf.cast(b, y.dtype),
                                  axis=list(range(2, y.shape.ndims)))
      y = y * tf.cast(w, y.dtype)
    return y, bias

  @abc.abstractmethod
  def custom_op(self, op, *args, **kwargs):
    """Double-dispatch: invokes a `visit_xxx` method on `op`."""


@six.add_metaclass(abc.ABCMeta)
class AffineLayer(SingleVerifiableLayer):
  """Layer that acts as an affine transform, e.g. linear or convolution."""

  @property
  def is_activation(self):
    return False

  @abc.abstractmethod
  def forward_prop(self, x, apply_bias=False, w_fn=None):
    """Applies the affine transform to a tensor.

    Args:
      x: Tensor of shape (num_targets, batch_size, input_shape...).
      apply_bias: whether to include the `b` contribution.
      w_fn: Optional elementwise preprocessing function to apply to `w`,
        for example `tf.abs`.

    Returns:
      Tensor of shape (num_targets, batch_size, output_shape...),
      containing  w x + b .
    """

  @abc.abstractmethod
  def backward_prop(self, y, w_fn=None):
    """Applies the transpose of the affine transform to a tensor.

    Args:
      y: Tensor of shape (num_targets, batch_size, output_shape...).
      w_fn: Optional elementwise preprocessing function to apply to `w`,
        for example `tf.abs`.

    Returns:
      Tensor of shape (num_targets, batch_size, input_shape...),
      containing  w^T y .
    """

  @abc.abstractmethod
  def backward_prop_bias(self, y):
    """Takes the scalar product of the bias with a tensor.

    Args:
      y: Tensor of shape (num_targets, batch_size, output_shape...).

    Returns:
      Tensor of shape (num_targets, batch_size),
      containing  b^T y .
    """

  @abc.abstractmethod
  def flatten(self):
    """Flattens the affine transform, materialising it as fully connected.

    Returns:
      w_flat:
        2D tensor of shape (input_size, output_size).
      b_flat:
        1D tensor of shape (output_size).
    """


class Conv(AffineLayer):
  """Wraps a convolutional layer."""

  def __init__(self, input_node, output_node, module, batch_norm=None,
               reshape=False):
    super(Conv, self).__init__(input_node, output_node, module,
                               batch_norm=batch_norm, reshape=reshape)
    self._w = module.w
    self._b = module.b if module.has_bias else None
    self._padding = module.padding
    self._strides = module.stride[1:-1]

  def forward_prop(self, x, apply_bias=False, w_fn=None):
    w = w_fn(self._w) if w_fn is not None else self._w
    y = common.convolution(x, tf.cast(w, x.dtype),
                           padding=self._padding, strides=self._strides)
    if apply_bias and self._b is not None:
      y += tf.cast(self._b, x.dtype)
    return y

  def backward_prop(self, y, w_fn=None):
    w = w_fn(self._w) if w_fn is not None else self._w
    return common.conv_transpose(y, tf.cast(w, y.dtype),
                                 result_shape=self.input_shape,
                                 padding=self._padding, strides=self._strides)

  def backward_prop_bias(self, y):
    if self._b is not None:
      return tf.reduce_sum(y * tf.cast(self._b, y.dtype),
                           axis=list(range(2, y.shape.ndims)))
    else:
      return tf.zeros(tf.shape(y)[:2], dtype=y.dtype)

  def flatten(self):
    return layer_utils.materialise_conv(
        self._w, self._b, input_shape=self.input_shape,
        padding=self._padding, strides=self._strides)

  def custom_op(self, op, *args, **kwargs):
    return op.visit_conv(self, self._w, self._b,
                         self._padding, self._strides, *args, **kwargs)


class Linear(AffineLayer):
  """Wraps a linear layer."""

  def __init__(self, input_node, output_node, module, batch_norm=None,
               reshape=False):
    super(Linear, self).__init__(input_node, output_node, module,
                                 batch_norm=batch_norm, reshape=reshape)
    self._w = module.w
    self._b = module.b if module.has_bias else None

  def forward_prop(self, x, apply_bias=False, w_fn=None):
    w = w_fn(self._w) if w_fn is not None else self._w
    y = tf.tensordot(x, tf.cast(w, x.dtype), axes=1)
    if apply_bias and self._b is not None:
      y += tf.cast(self._b, x.dtype)
    return y

  def backward_prop(self, y, w_fn=None):
    w = w_fn(self._w) if w_fn is not None else self._w
    return tf.tensordot(y, tf.transpose(tf.cast(w, y.dtype)), axes=1)

  def backward_prop_bias(self, y):
    if self._b is not None:
      return tf.tensordot(y, tf.cast(self._b, y.dtype), axes=1)
    else:
      return tf.zeros(tf.shape(y)[:2], dtype=y.dtype)

  def flatten(self):
    return self._w, self._b

  def custom_op(self, op, *args, **kwargs):
    return op.visit_linear(self, self._w, self._b, *args, **kwargs)

  def get_objective_weights(self, labels, target_strategy=None):
    """Elides the objective with this (final) linear layer."""
    assert self._b is not None, 'Last layer must have a bias.'
    if target_strategy is None:
      w, b = common.targeted_objective(self._w, self._b, labels)
    else:
      w, b = target_strategy.target_objective(self._w, self._b, labels)

    return ObjectiveWeights(w, b)


class AvgPool(AffineLayer):
  """Wraps an average-pool layer."""

  def __init__(self, input_node, output_node,
               kernel_shape, strides, reshape=False):
    super(AvgPool, self).__init__(input_node, output_node,
                                  module=None,
                                  reshape=reshape)
    self._kernel_shape = list(kernel_shape) if kernel_shape else None
    self._strides = list(strides) if strides else None

  @property
  def kernel_shape(self):
    return self._kernel_shape

  @property
  def strides(self):
    return self._strides

  def forward_prop(self, x, apply_bias=False, w_fn=None):
    return self._module(x)

  def backward_prop(self, y, w_fn=None):
    del w_fn
    return common.avgpool_transpose(y, result_shape=self.input_shape,
                                    kernel_shape=self.kernel_shape,
                                    strides=self.strides)

  def backward_prop_bias(self, y):
    return tf.zeros(tf.shape(y)[:2], dtype=y.dtype)

  def flatten(self):
    raise NotImplementedError()

  def custom_op(self, op, *args, **kwargs):
    return op.visit_avgpool(self, *args, **kwargs)


class MaxPool(SingleVerifiableLayer):
  """Wraps a max-pool layer."""

  def __init__(self, input_node, output_node,
               kernel_shape, strides, with_relu=False, reshape=False):
    super(MaxPool, self).__init__(input_node, output_node,
                                  module=None,
                                  reshape=reshape)
    self._kernel_shape = list(kernel_shape) if kernel_shape else None
    self._strides = list(strides) if strides else None
    self._with_relu = with_relu

  @property
  def kernel_shape(self):
    return self._kernel_shape

  @property
  def strides(self):
    return self._strides

  @property
  def with_relu(self):
    return self._with_relu

  @property
  def is_activation(self):
    return True

  def custom_op(self, op, *args, **kwargs):
    return op.visit_maxpool(self, *args, **kwargs)


class Activation(SingleVerifiableLayer):
  """Wraps an activation."""

  def __init__(self, input_node, output_node, module,
               reshape=False, parameters=None):
    super(Activation, self).__init__(input_node, output_node, module,
                                     reshape=reshape)
    self._activation = module.__name__  # Convert to string.
    self._parameters = parameters

  @property
  def is_activation(self):
    return True

  @property
  def activation(self):
    return self._activation

  @property
  def parameters(self):
    return self._parameters

  def custom_op(self, op, *args, **kwargs):
    return op.visit_activation(self, *args, **kwargs)


def create_verifiable_layer(input_node, output_node, module,
                            batch_norm=None, reshape=False,
                            parameters=None):
  """Returns an instance of `SingleVerifiableLayer` for the specified module."""
  if isinstance(module, snt.Conv2D) or isinstance(module, snt.Conv1D):
    return Conv(input_node, output_node, module, batch_norm, reshape)
  elif isinstance(module, snt.Linear):
    return Linear(input_node, output_node, module, batch_norm, reshape)
  else:
    if batch_norm is not None:
      raise ValueError('Cannot add a batch normalization layer to an '
                       'activation.')
    return Activation(input_node, output_node, module, reshape,
                      parameters=parameters)
