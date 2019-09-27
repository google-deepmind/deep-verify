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

"""Base test class for different verification strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from absl.testing import parameterized
from deep_verify.src import auto_verifier
from deep_verify.src.bounds import naive_bounds
import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


ImageLabel = collections.namedtuple('ImageLabel', ('image', 'label'))


class _TestNet(snt.AbstractModule):

  def __init__(self, num_classes, build_fn):
    super(_TestNet, self).__init__()
    self._num_classes = num_classes
    self._build_fn = build_fn

  @property
  def output_size(self):
    return self._num_classes

  def _build(self, image_batch):
    input_node = ibp.ModelInputWrapper(0)
    self._nodes_list = [input_node]
    self._nodes = {image_batch: input_node}
    self._node_inputs = {}
    self._fanouts = collections.Counter()

    outputs = self._build_fn(self._num_classes, self, image_batch)

    self._fanouts[self._nodes[outputs]] += 1
    self._outputs = outputs

    return outputs

  def append(self, node, outputs, *inputs):
    self._nodes_list.append(node)
    self._node_inputs[node] = inputs
    for x in inputs:
      self._fanouts[self._nodes[x]] += 1
    self._nodes[outputs] = node

  @property
  def output_module(self):
    self._ensure_is_connected()
    return self._nodes[self._outputs]

  def dependencies(self, node):
    return [self._nodes[inputs] for inputs in self._node_inputs[node]]

  @property
  def modules(self):
    self._ensure_is_connected()
    # Omit the (virtual) network input node.
    return self._nodes_list[1:]

  def fanout_of(self, node):
    self._ensure_is_connected()
    return self._fanouts[node]

  def propagate_bounds(self, input_bounds):
    self._nodes_list[0].output_bounds = input_bounds
    for node in self._nodes_list[1:]:
      node.propagate_bounds(*[self._nodes[inputs].output_bounds
                              for inputs in self._node_inputs[node]])


class AvgPool(snt.AbstractModule):
  """Wraps `tf.nn.avg_pool` as a callable object with properties."""

  def __init__(self, kernel_shape, strides):
    """Constructor.

    No padding is supported: `tf.nn.avg_pool` is invoked with `padding='VALID'`.

    Args:
      kernel_shape: Integer list of `[kernel_height, kernel_width]`.
      strides: Integer list of `[vertical_stride, horizontal_stride]`.
    """
    super(AvgPool, self).__init__()
    self._kernel_shape = list(kernel_shape)
    self._strides = list(strides)

  @property
  def kernel_shape(self):
    return self._kernel_shape

  @property
  def padding(self):
    return 'VALID'  # No padding is supported.

  @property
  def strides(self):
    return self._strides

  def _build(self, value):
    return tf.nn.avg_pool(value,
                          ksize=([1] + self._kernel_shape + [1]),
                          padding=self.padding,
                          strides=([1] + self._strides + [1]))


def add_layer(net, module, inputs, flatten=False, batch_norm=None):
  if flatten:
    reshape_module = snt.BatchFlatten()
    outputs = reshape_module(inputs)
    net.append(ibp.BatchReshapeWrapper(reshape_module,
                                       outputs.shape[1:].as_list()),
               outputs, inputs)
    inputs = outputs

  outputs = module(inputs)
  if isinstance(module, AvgPool):
    module.__name__ = 'avg_pool'
    parameters = {'ksize': [1] + module.kernel_shape + [1],
                  'padding': module.padding,
                  'strides': [1] + module.strides + [1]}
    net.append(ibp.IncreasingMonotonicWrapper(module, **parameters),
               outputs, inputs)
  elif isinstance(module, snt.Conv2D):
    net.append(ibp.LinearConv2dWrapper(module), outputs, inputs)
  elif isinstance(module, snt.Conv1D):
    net.append(ibp.LinearConv1dWrapper(module), outputs, inputs)
  elif isinstance(module, snt.Linear):
    net.append(ibp.LinearFCWrapper(module), outputs, inputs)
  else:
    net.append(ibp.IncreasingMonotonicWrapper(module), outputs, inputs)

  if batch_norm is not None:
    inputs = outputs
    outputs = batch_norm(inputs,
                         is_training=False, test_local_stats=False)
    net.append(ibp.BatchNormWrapper(batch_norm), outputs, inputs)

  return outputs


_inits = {'initializers': {'b': tf.truncated_normal_initializer()}}
_bn_inits = {'initializers': {
    'beta': tf.truncated_normal_initializer(),
    'gamma': tf.constant_initializer(0.7)
}}


def _linear_script(num_classes, net, layer_values):
  layer_values = add_layer(net, snt.Linear(13, **_inits), layer_values,
                           flatten=True)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(num_classes,
                                           **_inits), layer_values)
  return layer_values


def _conv_script(num_classes, net, layer_values):
  layer_values = add_layer(net, snt.Conv2D(3, kernel_shape=(2, 2),
                                           padding='VALID',
                                           **_inits), layer_values)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(11, **_inits), layer_values,
                           flatten=True)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(num_classes,
                                           **_inits), layer_values)
  return layer_values


def _conv_batchnorm_script(num_classes, net, layer_values):
  layer_values = add_layer(net, snt.Conv2D(3, kernel_shape=(2, 2),
                                           padding='VALID',
                                           use_bias=False
                                          ), layer_values,
                           batch_norm=snt.BatchNorm(scale=True, **_bn_inits))
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(11, use_bias=False
                                          ), layer_values,
                           flatten=True,
                           batch_norm=snt.BatchNorm(scale=True, **_bn_inits))
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(num_classes,
                                           **_inits), layer_values)
  return layer_values


def _avgpool_script(num_classes, net, layer_values):
  layer_values = add_layer(net, snt.Conv2D(3, kernel_shape=(2, 2),
                                           **_inits), layer_values)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, AvgPool(
      kernel_shape=(2, 2), strides=(1, 1)), layer_values)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, snt.Linear(num_classes,
                                           **_inits), layer_values,
                           flatten=True)
  return layer_values


def _avgpool_linear_script(num_classes, net, layer_values):
  layer_values = add_layer(net, snt.Conv2D(3, kernel_shape=(2, 2),
                                           **_inits), layer_values)
  layer_values = add_layer(net, tf.nn.relu, layer_values)
  layer_values = add_layer(net, AvgPool(
      kernel_shape=(2, 2), strides=(1, 1)), layer_values)
  layer_values = add_layer(net, snt.Linear(num_classes,
                                           **_inits), layer_values,
                           flatten=True)
  return layer_values


class DualFormulationTest(tf.test.TestCase, parameterized.TestCase):

  @abc.abstractmethod
  def _verification_strategy(self):
    pass

  def _num_classes(self):
    return 3

  def _batch_size(self):
    return 3

  def _image_data(self):
    image = tf.random_uniform(shape=(self._batch_size(), 5, 3, 2),
                              dtype=tf.float32)
    label = tf.random_uniform((self._batch_size(),),
                              maxval=self._num_classes(),
                              dtype=tf.int64)
    return ImageLabel(image=image, label=label)

  def _network(self, model):
    return _TestNet(self._num_classes(), self._script_fn(model))

  def _script_fn(self, model):
    return globals()['_' + model + '_script']

  def _verifiable_layer_builder(self, net):
    return auto_verifier.VerifiableLayerBuilder(net)

  def _apply_verification(self, model, objective_computation_config=None):
    image_data = self._image_data()
    net = self._network(model)
    net(image_data.image)

    # Bound propagation is performed on the graph representation.
    input_bounds = naive_bounds.input_bounds(image_data.image, delta=.1)
    boundprop_method = naive_bounds.NaiveBoundPropagation()
    boundprop_method.propagate_bounds(net, input_bounds)

    net_layers = self._verifiable_layer_builder(net).build_layers()
    grouped_layers = self._verification_strategy().group_layers(net_layers)

    dual_obj, _, project_duals_op, supporting_ops = (
        self._verification_strategy().create_duals_and_build_objective(
            grouped_layers,
            image_data.label,
            tf.get_variable,
            objective_computation_config=objective_computation_config))

    self.assertEqual((self._num_classes(), self._batch_size()),
                     tuple(dual_obj.shape.as_list()))

    # Smoke test: ensure that the verification calculation can run.
    init_op = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init_op)
      session.run(supporting_ops['init'])
      session.run(project_duals_op)
      session.run(dual_obj)

  def _build_objective(self, net, input_bounds, labels,
                       boundprop_method=naive_bounds.NaiveBoundPropagation()):
    """Invokes the verification formulation for the given network.

    Args:
      net: `_TestNet` specifying network graph.
      input_bounds: Bounds for the network inputs.
      labels: 1D integer tensor of shape (batch_size) of labels for each
        input example.
      boundprop_method: Specifies the method used to propagate bounds.

    Returns:
      dual_obj: 2D tensor of shape (num_classes, batch_size) containing
        dual objective values for each (class, example).
      dual_var_lists: Nested list of 3D tensors of shape
        (num_classes, batch_size, layer_size) and 5D tensors of shape
        (num_classes, batch_size, height, width, channels)
        containing Lagrange multipliers for the layers' calculations.
        This has the same length as `verifiable_layers`, and typically each
        entry is a singleton list with the dual variable for that layer.
        ResNet blocks' entries have instead the structure
        [[left-sub-duals], [right-sub-duals], overall-dual].
    """
    net(input_bounds.nominal)  # Connect the network to the graph.
    boundprop_method.propagate_bounds(net, input_bounds)

    net_layers = self._verifiable_layer_builder(net).build_layers()
    grouped_layers = self._verification_strategy().group_layers(net_layers)

    dual_obj, dual_var_lists, _, _ = (
        self._verification_strategy().create_duals_and_build_objective(
            grouped_layers, labels, tf.get_variable, margin=1.))
    return dual_obj, dual_var_lists

  def _expected_input_bounds(self, input_data, delta):
    return tf.maximum(input_data-delta, 0.), tf.minimum(input_data+delta, 1.)

  def _assert_dual_objective_close(self, expected_objective,
                                   dual_obj, image_data):
    neq = tf.not_equal(
        tf.expand_dims(tf.range(self._num_classes(), dtype=tf.int64), axis=1),
        image_data.label)
    expected_objective = tf.where(neq, expected_objective,
                                  -tf.ones_like(expected_objective))

    init_op = tf.global_variables_initializer()
    with self.test_session() as session:
      session.run(init_op)
      expected_objective_val, dual_obj_val = session.run([expected_objective,
                                                          dual_obj])

    tol = 1e-6
    self.assertAllClose(expected_objective_val, dual_obj_val,
                        atol=tol, rtol=tol)
