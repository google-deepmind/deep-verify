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

"""Automatic construction of verifiable layers from a Sonnet module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deep_verify.src.layers import layers
import interval_bound_propagation as ibp
import sonnet as snt


class NotVerifiableError(Exception):
  """Module's graph contains features that do not map to verification layers."""


class VerifiableLayerBuilder(object):
  """Constructs verifiable layers from a Sonnet module."""

  def __init__(self, network):
    """Constructor.

    Args:
      network: `NetworkBuilder` containing network with propagated bounds.
    """
    super(VerifiableLayerBuilder, self).__init__()
    self._network = network

  def build_layers(self):
    """Builds the verifiable layers.

    Returns:
      List of `SingleVerifiableLayer` for the module.

    Raises:
      NotVerifiableError: on invalid layer arrangement.
    """
    backstop_node, known_fanout, verifiable_layers, reshape = (
        self._build_layers_rec(self._network.output_module))
    if (not isinstance(backstop_node, ibp.ModelInputWrapper) or
        self._network.fanout_of(backstop_node) != known_fanout):
      raise NotVerifiableError('Invalid connectivity')
    if reshape:
      raise NotVerifiableError('Cannot end with a reshape operation')
    return self._fuse(verifiable_layers)

  def _build_layers_rec(self, node, known_fanout=1, batchnorm_node=None):
    """Builds verifiable layers leading up to the given layer output.

    The list is constructed by navigating the layers in reverse order,
    stopping either when the module's original inputs are reached,
    or (for within a ResNet block) when a layer is encountered that has
    outputs not processed by this navigation.

    Args:
      node: Layer output, up to which to build verifiable layers.
      known_fanout: Number of immediate outputs of `layer_tensor` that have
        already been processed by the caller.
        This is typically 1, but sub-classes may invoke with 2 (or possibly
        greater) where the network contains branches.
      batchnorm_node: The BatchNorm's ConnectedSubgraph object if
        `layer_tensor` is the input to a BatchNorm layer, otherwise None.

    Returns:
      backstop_node: Node, typically the `ibp.ModelInputWrapper`, at which we
        stopped backtracking.
      known_fanout: Number of immediate outputs of `input_tensor` that were
        processed in this call.
        This is typically 1, but overrides may return 2 (or possibly greater)
        in the presence of branched architectures.
      verifiable_layers: List of `SingleVerifiableLayer` whose final element's
        output is `outputs`.
      reshape: Whether the final element of `verifiable_layers` is followed by
        a reshape operation.

    Raises:
      NotVerifiableError: on invalid layer arrangement.
    """
    if (isinstance(node, ibp.ModelInputWrapper) or
        self._network.fanout_of(node) != known_fanout):
      # Reached the inputs (or start of the enclosing ResNet block).
      # No more layers to construct.
      if batchnorm_node:
        raise NotVerifiableError('Cannot begin with batchnorm')
      return node, known_fanout, [], False

    elif (isinstance(node, ibp.IncreasingMonotonicWrapper) and
          node.module.__name__ == 'identity'):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      return self._build_layers_rec(input_node, batchnorm_node=batchnorm_node)

    elif (isinstance(node, ibp.IncreasingMonotonicWrapper) and
          node.module.__name__ == 'avg_pool'):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      input_tensor, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))

      # Construct the AvgPool layer.
      if batchnorm_node:
        raise NotVerifiableError('AvgPool cannot have batchnorm')
      if node.parameters['padding'] == 'SAME':
        raise ValueError('"SAME" padding is not supported.')
      verifiable_layers.append(layers.AvgPool(
          input_node,
          node,
          kernel_shape=node.parameters['ksize'][1:-1],
          strides=node.parameters['strides'][1:-1],
          reshape=reshape))
      return input_tensor, known_fanout, verifiable_layers, False

    elif (isinstance(node, ibp.IncreasingMonotonicWrapper) and
          node.module.__name__ == 'reduce_mean'):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      input_tensor, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))

      # Construct the AvgPool layer.
      if batchnorm_node:
        raise NotVerifiableError('AvgPool cannot have batchnorm')
      verifiable_layers.append(layers.AvgPool(
          input_node,
          node,
          kernel_shape=None,
          strides=None,
          reshape=reshape))
      return input_tensor, known_fanout, verifiable_layers, False

    elif (isinstance(node, ibp.IncreasingMonotonicWrapper) and
          node.module.__name__ == 'max_pool'):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      input_tensor, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))

      # Construct the MaxPool layer.
      if batchnorm_node:
        raise NotVerifiableError('MaxPool cannot have batchnorm')
      if node.parameters['padding'] == 'SAME':
        raise ValueError('"SAME" padding is not supported.')
      verifiable_layers.append(layers.MaxPool(
          input_node,
          node,
          kernel_shape=node.parameters['ksize'][1:-1],
          strides=node.parameters['strides'][1:-1],
          reshape=reshape))
      return input_tensor, known_fanout, verifiable_layers, False

    elif (isinstance(node, ibp.IncreasingMonotonicWrapper) and
          node.module.__name__ == 'reduce_max'):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      input_tensor, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))

      # Construct the MaxPool layer.
      if batchnorm_node:
        raise NotVerifiableError('MaxPool cannot have batchnorm')
      verifiable_layers.append(layers.MaxPool(
          input_node,
          node,
          kernel_shape=None,
          strides=None,
          reshape=reshape))
      return input_tensor, known_fanout, verifiable_layers, False

    elif isinstance(node.module, snt.BatchNorm):
      # Construct the previous layer with batchnorm.
      if batchnorm_node:
        raise NotVerifiableError('Cannot have consecutive batchnorms')
      input_node, = self._network.dependencies(node)
      return self._build_layers_rec(input_node, batchnorm_node=node)

    elif isinstance(node.module, snt.BatchReshape):
      # Recursively build all preceding layers.
      input_node, = self._network.dependencies(node)
      backstop_node, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))
      if batchnorm_node:
        raise NotVerifiableError('Reshape cannot have batchnorm')
      return backstop_node, known_fanout, verifiable_layers, True

    else:
      # Recursively build all preceding layers.
      input_nodes = self._network.dependencies(node)
      if len(input_nodes) != 1:
        raise NotVerifiableError('Unary operation expected')
      input_node, = input_nodes
      backstop_node, known_fanout, verifiable_layers, reshape = (
          self._build_layers_rec(input_node))

      # Construct the layer.
      verifiable_layers.append(layers.create_verifiable_layer(
          input_node,
          batchnorm_node or node,
          node.module,
          batch_norm=(batchnorm_node.module if batchnorm_node else None),
          reshape=reshape,
          parameters=(node.parameters
                      if isinstance(node, ibp.IncreasingMonotonicWrapper)
                      else None),
      ))

      return backstop_node, known_fanout, verifiable_layers, False

  def _fuse(self, verifiable_layers):
    """Performs fusion of certain layer pairs."""
    fused_layers = []
    idx = 0
    while idx < len(verifiable_layers):

      if (idx+2 <= len(verifiable_layers) and
          isinstance(verifiable_layers[idx], layers.MaxPool) and
          isinstance(verifiable_layers[idx+1], layers.Activation) and
          verifiable_layers[idx+1].activation == 'relu'):
        # Fuse maxpool with relu.
        original = verifiable_layers[idx]
        fused_layers.append(layers.MaxPool(original.input_node,
                                           original.output_node,
                                           kernel_shape=original.kernel_shape,
                                           strides=original.strides,
                                           with_relu=True,
                                           reshape=original.reshape))
        idx += 2

      else:
        fused_layers.append(verifiable_layers[idx])
        idx += 1

    return fused_layers
