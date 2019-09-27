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

"""Tests for `auto_verifier`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from deep_verify.src import auto_verifier
from deep_verify.src.layers import layers
import interval_bound_propagation as ibp
import sonnet as snt
import tensorflow as tf


class _BatchNorm(snt.BatchNorm):

  def _build(self, input_batch):
    return super(_BatchNorm, self)._build(input_batch,
                                          is_training=False,
                                          test_local_stats=False)


class AutoVerifierTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(AutoVerifierTest, self).setUp()
    self._inputs = tf.placeholder(dtype=tf.float32, shape=(11, 28, 28, 3))

  def test_empty_network(self):
    module = snt.Sequential([])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertEmpty(v_layers)

  def test_standalone_conv_module(self):
    module = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)

    self.assertIsInstance(v_layers[0], layers.Conv)
    self.assertIs(module, v_layers[0].module)
    self.assertIsInstance(v_layers[0].input_node, ibp.ModelInputWrapper)

    self.assertIs(v_layers[0].output_node, network.output_module)

  def test_flatten_and_linear(self):
    linear = snt.Linear(23)
    module = snt.Sequential([
        snt.BatchFlatten(),
        linear,
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)

    self.assertIsInstance(v_layers[0], layers.Linear)
    self.assertIs(linear, v_layers[0].module)
    self.assertEqual([2352], v_layers[0].input_node.shape)

    self.assertIs(v_layers[0].output_node, network.output_module)

  def test_pointless_reshape(self):
    linear = snt.Linear(23)
    module = snt.Sequential([
        snt.BatchFlatten(),
        linear,
        snt.BatchFlatten(),
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)

    self.assertIsInstance(v_layers[0], layers.Linear)
    self.assertIs(linear, v_layers[0].module)
    self.assertEqual([2352], v_layers[0].input_node.shape)

    self.assertIs(v_layers[0].output_node, network.output_module)

  def test_unrecognised_calculation_rejected(self):
    class InvalidModule(snt.AbstractModule):

      def _build(self, inputs):
        module = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
        return module(2 * inputs)

    module = InvalidModule()

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    with self.assertRaises(auto_verifier.NotVerifiableError):
      _ = auto_verifier.VerifiableLayerBuilder(network).build_layers()

  def test_unrecognised_trailing_calculation_rejected(self):
    class InvalidModule(snt.AbstractModule):

      def _build(self, inputs):
        module = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
        return 2 * module(inputs)

    module = InvalidModule()

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    with self.assertRaises(auto_verifier.NotVerifiableError):
      _ = auto_verifier.VerifiableLayerBuilder(network).build_layers()

  @parameterized.named_parameters(('relu', tf.nn.relu),
                                  ('sig', tf.nn.sigmoid),
                                  ('tanh', tf.nn.tanh),
                                  ('elu', tf.nn.elu))
  def test_stack_with_snt_activation(self, activation_fn):
    conv = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
    linear = snt.Linear(23)
    module = snt.Sequential([
        conv,
        snt.Module(activation_fn),
        snt.BatchFlatten(),
        linear,
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 3)

    self.assertIsInstance(v_layers[0], layers.Conv)
    self.assertIs(conv, v_layers[0].module)
    self.assertIsInstance(v_layers[0].input_node, ibp.ModelInputWrapper)

    self.assertIsInstance(v_layers[1], layers.Activation)
    self.assertEqual(activation_fn.__name__, v_layers[1].activation)
    self.assertIs(v_layers[0].output_node, v_layers[1].input_node)

    self.assertIsInstance(v_layers[2], layers.Linear)
    self.assertIs(linear, v_layers[2].module)

    self.assertIs(v_layers[2].output_node, network.output_module)

  @parameterized.named_parameters(('relu', tf.nn.relu),
                                  ('sig', tf.nn.sigmoid),
                                  ('tanh', tf.nn.tanh),
                                  ('elu', tf.nn.elu),)
  def test_stack_with_tf_activation(self, activation_fn):
    conv = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
    linear = snt.Linear(23)
    module = snt.Sequential([
        conv,
        activation_fn,
        snt.BatchFlatten(),
        linear
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 3)

    self.assertIsInstance(v_layers[0], layers.Conv)
    self.assertIs(conv, v_layers[0].module)
    self.assertIsInstance(v_layers[0].input_node, ibp.ModelInputWrapper)

    self.assertIsInstance(v_layers[1], layers.Activation)
    self.assertEqual(activation_fn.__name__, v_layers[1].activation)
    self.assertIs(v_layers[0].output_node, v_layers[1].input_node)

    self.assertIsInstance(v_layers[2], layers.Linear)
    self.assertIs(linear, v_layers[2].module)

    self.assertIs(v_layers[2].output_node, network.output_module)

  def test_batchnorm(self):
    conv = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
    batchnorm = _BatchNorm()
    module = snt.Sequential([
        conv,
        batchnorm,
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)

    self.assertIsInstance(v_layers[0], layers.Conv)
    self.assertIs(conv, v_layers[0].module)
    self.assertIs(batchnorm, v_layers[0].batch_norm)
    self.assertIsInstance(v_layers[0].input_node, ibp.ModelInputWrapper)

    self.assertIs(v_layers[0].output_node, network.output_module)

  def test_consecutive_batchnorm_rejected(self):
    module = snt.Sequential([
        snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID'),
        _BatchNorm(),
        _BatchNorm(),
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    with self.assertRaises(auto_verifier.NotVerifiableError):
      _ = auto_verifier.VerifiableLayerBuilder(network).build_layers()

  def test_leading_batchnorm_rejected(self):
    module = snt.Sequential([
        _BatchNorm(),
        snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID'),
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    with self.assertRaises(auto_verifier.NotVerifiableError):
      _ = auto_verifier.VerifiableLayerBuilder(network).build_layers()

  def test_tolerates_identity(self):
    conv = snt.Conv2D(output_channels=5, kernel_shape=3, padding='VALID')
    linear = snt.Linear(23)
    module = snt.Sequential([
        tf.identity,
        conv,
        tf.identity,
        tf.nn.relu,
        tf.identity,
        snt.BatchFlatten(),
        tf.identity,
        linear,
        tf.identity,
    ])

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 3)

    self.assertIsInstance(v_layers[0], layers.Conv)
    self.assertIs(conv, v_layers[0].module)

    self.assertIsInstance(v_layers[1], layers.Activation)
    self.assertEqual('relu', v_layers[1].activation)

    self.assertIsInstance(v_layers[2], layers.Linear)
    self.assertIs(linear, v_layers[2].module)

  def test_leaky_relu(self):
    module = lambda x: tf.nn.leaky_relu(x, alpha=0.3)

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)
    self.assertIsInstance(v_layers[0], layers.Activation)
    self.assertEqual('leaky_relu', v_layers[0].activation)
    self.assertLen(v_layers[0].parameters, 1)
    self.assertAllClose(0.3, v_layers[0].parameters['alpha'])

  def test_avgpool(self):
    def module(inputs):
      return tf.nn.avg_pool(inputs, ksize=(3, 3),
                            padding='VALID', strides=(2, 2))

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)
    self.assertIsInstance(v_layers[0], layers.AvgPool)
    self.assertEqual([3, 3], v_layers[0].kernel_shape)
    self.assertEqual([2, 2], v_layers[0].strides)

  def test_avgpool_global(self):
    def module(inputs):
      return tf.reduce_mean(inputs, axis=(1, 2))

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)
    self.assertIsInstance(v_layers[0], layers.AvgPool)
    self.assertIs(None, v_layers[0].kernel_shape)

  @parameterized.named_parameters(('plain', False),
                                  ('relu', True))
  def test_maxpool(self, with_relu):
    def module(inputs):
      outputs = tf.nn.max_pool(inputs, ksize=(3, 3),
                               padding='VALID', strides=(2, 2))
      if with_relu:
        outputs = tf.nn.relu(outputs)
      return outputs

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)
    self.assertIsInstance(v_layers[0], layers.MaxPool)
    self.assertEqual([3, 3], v_layers[0].kernel_shape)
    self.assertEqual([2, 2], v_layers[0].strides)
    self.assertEqual(with_relu, v_layers[0].with_relu)

  @parameterized.named_parameters(('plain', False),
                                  ('relu', True))
  def test_maxpool_global(self, with_relu):
    def module(inputs):
      outputs = tf.reduce_max(inputs, axis=(1, 2))
      if with_relu:
        outputs = tf.nn.relu(outputs)
      return outputs

    network = ibp.VerifiableModelWrapper(module)
    network(self._inputs)

    v_layers = auto_verifier.VerifiableLayerBuilder(network).build_layers()

    self.assertLen(v_layers, 1)
    self.assertIsInstance(v_layers[0], layers.MaxPool)
    self.assertIs(None, v_layers[0].kernel_shape)
    self.assertEqual(with_relu, v_layers[0].with_relu)


if __name__ == '__main__':
  tf.test.main()
