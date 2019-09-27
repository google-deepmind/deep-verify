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

"""Verifies a pre-trained MNIST model.

Usage for pre-training and verifying:
    python verify.py

Alternatively, the classifier may be pre-trained with IBP:
    python interval_bound_propagation/examples/train.py
        --output_dir=/tmp/ibp_model --num_steps=60001
and then verified:
    python verify.py --pretrained_model_path=/tmp/ibp_model/model-60000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import deep_verify
import interval_bound_propagation as ibp
import tensorflow as tf


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'mnist', 'Dataset (either "mnist" or "cifar10")')
flags.DEFINE_string('model', 'tiny', 'Model size')
flags.DEFINE_string('pretrained_model_path', '',
                    'Optional path from which to restore pre-trained model.')

flags.DEFINE_integer('pretrain_batch_size', 200, 'Batch size for pre-training.')
flags.DEFINE_integer('test_batch_size', 200, 'Batch size for nominal testing.')
flags.DEFINE_integer('pretrain_steps', 10001, 'Number of pre-training steps.')
flags.DEFINE_integer('test_every_n', 2000,
                     'Number of steps between testing iterations.')
flags.DEFINE_string('learning_rate', '1e-3,1e-4@5000,1e-5@7500',
                    'Learning rate schedule of the form: '
                    'initial_learning_rate[,learning:steps]*. E.g., "1e-3" or '
                    '"1e-3,1e-4@5000,1e-5@7500".')

flags.DEFINE_float('epsilon', .02, 'Perturbation radius.')
flags.DEFINE_integer('verification_batch_size', 100,
                     'Batch size for verification.')
flags.DEFINE_integer('verification_steps', 2001,
                     'Number of steps of dual variable optimization per batch.')
flags.DEFINE_float('dual_learning_rate', 1e-3,
                   'Learning rate for verification.')


def layers(model_size):
  """Returns the layer specification for a given model name."""
  if model_size == 'tiny':
    return (
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'small':
    return (
        ('conv2d', (4, 4), 16, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'medium':
    return (
        ('conv2d', (3, 3), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 64, 'VALID', 2),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'))
  elif model_size == 'large':
    return (
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('linear', 200),
        ('activation', 'relu'))
  else:
    raise ValueError('Unknown model: "{}"'.format(model_size))


def pretraining_graph(classifier,
                      data_train, train_batch_size, train_randomize_fn,
                      step, learning_rate):
  """Constructs the TensorFlow graph for pre-training the model."""
  train_data = ibp.build_dataset(data_train, batch_size=train_batch_size,
                                 sequential=False)
  if train_randomize_fn is not None:
    train_data = train_data._replace(image=train_randomize_fn(train_data.image))

  train_logits = classifier(train_data.image, is_training=True)
  train_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=train_data.label, logits=train_logits))

  learning_rate = ibp.parse_learning_rate(step, learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(train_loss, step)

  return train_op, train_loss


def nominal_accuracy_graph(classifier, data_test, test_batch_size):
  """Constructs the TensorFlow graph for computing the model's accuracy."""
  # Test using while loop.
  test_set_size = len(data_test[0])
  if test_set_size % test_batch_size != 0:
    logging.warn('Test set (size %d) is not a whole number of batches '
                 '(size %d). Some examples at the end of the test set will be '
                 'skipped.', test_set_size, test_batch_size)
  num_test_batches = test_set_size // test_batch_size

  def cond(i, *unused_args):
    return i < num_test_batches

  def body(i, total_test_accuracy):
    """Compute the sum of all metrics."""
    test_data = ibp.build_dataset(data_test,
                                  batch_size=test_batch_size,
                                  sequential=True)
    test_logits = classifier(test_data.image, is_training=False)
    test_correct = tf.equal(test_data.label, tf.argmax(test_logits, axis=1))
    test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))

    return i + 1, total_test_accuracy + test_accuracy

  i = tf.zeros(shape=(), dtype=tf.int32)
  total_test_accuracy = tf.zeros(shape=(), dtype=tf.float32)
  i, total_test_accuracy = tf.while_loop(
      cond,
      body,
      loop_vars=[i, total_test_accuracy],
      back_prop=False,
      parallel_iterations=1)
  total_count = tf.cast(i, tf.float32)
  test_accuracy = total_test_accuracy / total_count

  return test_accuracy


def verification_graph(classifier, epsilon,
                       data_test, test_batch_size,
                       learning_rate):
  """Constructs the TensorFlow graph for the verification computation."""
  test_data_live = ibp.build_dataset(data_test,
                                     batch_size=test_batch_size,
                                     sequential=True)
  tf.contrib.framework.nest.map_structure(
      lambda x: x.set_shape([test_batch_size] + x.shape[1:]),
      test_data_live)
  test_data, get_next_batch_op = deep_verify.with_explicit_update(
      test_data_live)

  net_builder = ibp.VerifiableModelWrapper(classifier)
  net_builder(test_data.image)

  input_bounds = deep_verify.input_bounds(test_data.image, epsilon)
  boundprop_method = deep_verify.NaiveBoundPropagation()
  boundprop_method.propagate_bounds(net_builder, input_bounds)

  verifiable_layers = deep_verify.VerifiableLayerBuilder(
      net_builder).build_layers()

  formulation = deep_verify.StandardDualFormulation()
  grouped_layers = formulation.group_layers(verifiable_layers)
  dual_verification = deep_verify.DualVerification(formulation, grouped_layers)

  dual_obj = dual_verification(test_data.label,
                               num_batches=1, current_batch=0, margin=1.)
  dual_loss = tf.reduce_mean(dual_obj)
  verified_correct = tf.reduce_max(dual_obj, axis=0) < 0
  verified_accuracy = tf.reduce_mean(tf.cast(verified_correct, tf.float32))

  dual_optimizer = tf.train.AdamOptimizer(learning_rate)
  dual_train_op = dual_optimizer.minimize(
      dual_loss, var_list=dual_verification.get_variables())

  get_next_batch_op = tf.group([
      get_next_batch_op,
      tf.variables_initializer(dual_verification.get_variables()),
      tf.variables_initializer(dual_optimizer.variables())])

  return get_next_batch_op, dual_train_op, verified_accuracy


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  num_classes = 10
  if FLAGS.dataset == 'mnist':
    data_train, data_test = tf.keras.datasets.mnist.load_data()
  else:
    assert FLAGS.dataset == 'cifar10', (
        'Unknown dataset "{}"'.format(FLAGS.dataset))
    data_train, data_test = tf.keras.datasets.cifar10.load_data()
    data_train = (data_train[0], data_train[1].flatten())
    data_test = (data_test[0], data_test[1].flatten())

  # Base classifier network.
  original_classifier = ibp.DNN(num_classes, layers(FLAGS.model))
  classifier = original_classifier
  if FLAGS.dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    classifier = ibp.add_image_normalization(original_classifier, mean, std)

  if FLAGS.dataset == 'cifar10':
    def train_randomize_fn(image):
      return ibp.randomize(image, (32, 32, 3), expand_shape=(40, 40, 3),
                           crop_shape=(32, 32, 3), vertical_flip=True)
  else:
    train_randomize_fn = None

  step = tf.train.get_or_create_global_step()

  train_op, train_loss = pretraining_graph(
      classifier,
      data_train, FLAGS.pretrain_batch_size, train_randomize_fn,
      step, FLAGS.learning_rate)

  test_accuracy = nominal_accuracy_graph(
      classifier,
      data_test, FLAGS.test_batch_size)

  if FLAGS.pretrained_model_path:
    saver = tf.train.Saver(original_classifier.get_variables())

  # Accompanying verification graph.
  get_next_batch_op, dual_train_op, verified_accuracy = verification_graph(
      classifier, FLAGS.epsilon,
      data_test, FLAGS.verification_batch_size,
      FLAGS.dual_learning_rate)
  test_set_size = len(data_test[0])
  if test_set_size % FLAGS.verification_batch_size != 0:
    logging.warn('Test set (size %d) is not a whole number of batches '
                 '(size %d). Some examples at the end of the test set will be '
                 'skipped.', test_set_size, FLAGS.verification_batch_size)
  num_batches = test_set_size // FLAGS.verification_batch_size

  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  with tf.train.SingularMonitoredSession(config=tf_config) as sess:

    if FLAGS.pretrained_model_path:
      print('Loading pre-trained model')
      saver.restore(sess._tf_sess(),  # pylint: disable=protected-access,
                    FLAGS.pretrained_model_path)
      test_accuracy_val = sess.run(test_accuracy)
      print('Loaded model:  Test accuracy {:.2f}%'.format(
          test_accuracy_val*100))

    else:
      print('Pre-training')
      for _ in range(FLAGS.pretrain_steps):
        iteration, train_loss_val, _ = sess.run([step, train_loss, train_op])
        if iteration % FLAGS.test_every_n == 0:
          test_accuracy_val = sess.run(test_accuracy)
          print('Step {}:  Test accuracy {:.2f}%  Train loss {:.4f}'.format(
              iteration, test_accuracy_val*100, train_loss_val))

    print('Verification')
    verified_accuracy_total = 0.
    for batch in range(num_batches):
      sess.run(get_next_batch_op)
      for iteration in range(FLAGS.verification_steps):
        sess.run(dual_train_op)
        if iteration % 200 == 0:
          verified_accuracy_val = sess.run(verified_accuracy)
          print('Batch {}:  Verified accuracy {:.2f}%'.format(
              batch, verified_accuracy_val*100))
      verified_accuracy_total += verified_accuracy_val
    print('Whole dataset:  Verified accuracy {:.2f}%'.format(
        (verified_accuracy_val/num_batches)*100))


if __name__ == '__main__':
  app.run(main)
