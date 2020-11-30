# Copyright 2016 Google Inc.
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
# limitations under the License
# ==============================================================================
"""
Learning 2 Learn problems.

在原来l2o-dm的代码的基础上加入了rastirgin函数，要使用必须要在代码文件夹中加入Rastrigin_data
文件夹以提供数据。

三个重要参数：batch_size=1280，num_dims=2或10，mode='train'或'test'。

这里batch_size必须在train和test都取1280，因为在test中为了简化代码，直接把test原数据复制10遍放在tensor中，这样执行一次
相当于就对每组数据做了10次test，于是batch_size在这里就必须为1280。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import sys

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import numpy as np
#这里增加import了numpy
import pdb
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
from vgg16 import VGG16


_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}

def simple():
  """Simple problem: f(x) = x^2."""

  def build():
    """Builds loss graph."""
    x = tf.get_variable(
        "x",
        shape=[],
        dtype=tf.float32,
        initializer=tf.ones_initializer())
    return tf.square(x, name="x_squared")

  return build


def simple_multi_optimizer(num_dims=2):
  """Multidimensional simple problem."""

  def get_coordinate(i):
    return tf.get_variable("x_{}".format(i),
                           shape=[],
                           dtype=tf.float32,
                           initializer=tf.ones_initializer())

  def build():
    coordinates = [get_coordinate(i) for i in xrange(num_dims)]
    x = tf.concat([tf.expand_dims(c, 0) for c in coordinates], 0)
    return tf.reduce_sum(tf.square(x, name="x_squared"))

  return build


def quadratic(batch_size=128, num_dims=10, stddev=0.01, dtype=tf.float32):
  """Quadratic problem: f(x) = ||Wx - y||."""

  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    return tf.reduce_mean(tf.reduce_sum((product - y) ** 2, 1))

  return build

def rastrigin(batch_size=1280, num_dims=2, mode=None, alpha=10, stddev=1, dtype=tf.float32):
  #这里batch_size必须在train和test都取1280，因为在test中为了简化代码，直接把test原数据复制10遍放在tensor中，这样执行一次
  #相当于就对每组数据做了10次test，于是batch_size在这里就必须为1280。
  def loadrasdata(condition,n,name):
    if condition == 'train':
      if n == 2:
        if name == 'a':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_a_2x2.npy")))
        elif name == 'b':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_b_2x1.npy")))
        elif name == 'c':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_c_2x1.npy")))

      elif n == 10:
        if name == 'a':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_a_10x10.npy")))
        elif name == 'b':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_b_10x1.npy")))
        elif name == 'c':
          return tf.to_float(tf.convert_to_tensor(np.load("Rastrigin_data/train_c_10x1.npy")))

    elif condition == 'test':
      if n == 2:
        #这里通过触发广播机制使原始读入的test数据复制多遍填满指定大小的tensor
        if name == 'a':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_a_2x2.npy"),[10,1,1])))
        elif name == 'b':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_b_2x1.npy"),[10,1,1])))
        elif name == 'c':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_c_2x1.npy"),[10,1,1])))

      elif n == 10:
        #这里通过触发广播机制使原始读入的test数据复制多遍填满指定大小的tensor
        if name == 'a':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_a_10x10.npy"),[10,1,1])))
        elif name == 'b':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_b_10x1.npy"),[10,1,1])))
        elif name == 'c':
          return tf.to_float(tf.convert_to_tensor(np.tile(np.load("Rastrigin_data/test_c_10x1.npy"),[10,1,1])))
  
  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims, 1],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    aq = tf.get_variable("aq",
                        dtype=dtype,
                        initializer=loadrasdata(mode,num_dims,'a'),
                        trainable=False)
    bq = tf.get_variable("bq",
                        dtype=dtype,
                        initializer=loadrasdata(mode,num_dims,'b'),
                        trainable=False)
    cq = tf.get_variable("cq",
                        dtype=dtype,
                        initializer=loadrasdata(mode,num_dims,'c'),
                        trainable=False)
    #here!

    ras_norm=tf.norm(tf.matmul(aq,x)-bq,ord=2,axis=[-2,-1])
    
    import math
    cqTcos=tf.matmul(tf.transpose(cq,perm=[0,2,1]),tf.cos(2*math.pi*x))

    return tf.reduce_mean(0.5*(ras_norm**2)-alpha*cqTcos+alpha*num_dims)

  return build

def lasso(batch_size=0, num_dims_m=0, num_dims_n=0, stddev=0.01, lamada=1.0, dtype=tf.float32):
  """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada *||x||1."""
  print("=" * 100)
  print("LASSO: BS={} m={} n={}".format(batch_size, num_dims_m, num_dims_n))
  print("=" * 100)
  def build():
    """Builds loss graph."""

    # Trainable variable.
    x = tf.get_variable(
        "x",
        shape=[batch_size, num_dims_n],
        dtype=dtype,
        initializer=tf.random_normal_initializer(stddev=stddev))

    # Non-trainable variables.
    w = tf.get_variable("w",
                        shape=[batch_size, num_dims_m, num_dims_n],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)
    y = tf.get_variable("y",
                        shape=[batch_size, num_dims_m],
                        dtype=dtype,
                        initializer=tf.random_uniform_initializer(),
                        trainable=False)

    product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
    left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
    other_term = lamada * tf.norm(x, ord=1)
    result = tf.reduce_mean(left_term + other_term)
    return result
  return build

# def lasso(batch_size=128, num_dims_m=5, num_dims_n=10, stddev=0.01, lamada=1.0, dtype=tf.float32):
#   """lasso problem: f(x) = 0.5*||Wx - y||2 + lamada||x||1."""

#   def build():
#     """Builds loss graph."""

#     # Trainable variable.
#     x = tf.get_variable(
#         "x",
#         shape=[batch_size, num_dims_n],
#         dtype=dtype,
#         initializer=tf.random_normal_initializer(stddev=stddev))

#     # Non-trainable variables.
#     w = tf.get_variable("w",
#                         shape=[batch_size, num_dims_m, num_dims_n],
#                         dtype=dtype,
#                         initializer=tf.random_uniform_initializer(),
#                         trainable=False)
#     y = tf.get_variable("y",
#                         shape=[batch_size, num_dims_m],
#                         dtype=dtype,
#                         initializer=tf.random_uniform_initializer(),
#                         trainable=False)

#     product = tf.squeeze(tf.matmul(w, tf.expand_dims(x, -1)))
#     left_term = 0.5 * tf.reduce_sum((product - y) ** 2, 1)
#     other_term = lamada * tf.norm(x, ord=1)
#     result = tf.reduce_mean(left_term + other_term)
#     return result

#   return build


def ensemble(problems, weights=None):
  """Ensemble of problems.

  Args:
    problems: List of problems. Each problem is specified by a dict containing
        the keys 'name' and 'options'.
    weights: Optional list of weights for each problem.

  Returns:
    Sum of (weighted) losses.

  Raises:
    ValueError: If weights has an incorrect length.
  """
  if weights and len(weights) != len(problems):
    raise ValueError("len(weights) != len(problems)")

  build_fns = [getattr(sys.modules[__name__], p["name"])(**p["options"])
               for p in problems]

  def build():
    loss = 0
    for i, build_fn in enumerate(build_fns):
      with tf.variable_scope("problem_{}".format(i)):
        loss_p = build_fn()
        if weights:
          loss_p *= weights[i]
        loss += loss_p
    return loss

  return build


def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)

def mnist(layers,  # pylint: disable=invalid-name
          activation="sigmoid",
          batch_size=128,
          mode="train",
          init="normal"):
  """Mnist classification with a multi-layer perceptron."""

  if init=="uniform" or init=="hessian":
      initializers = {
          "w": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
          "b": tf.random_uniform_initializer(minval=-0.0196, maxval=0.0196, seed=None),
      }
  else:
      initializers = _nn_initializers


  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

  # Network.
  mlp = snt.nets.MLP(list(layers) + [10],
                     activation=activation_op,
                     initializers=initializers)
  network = snt.Sequential([snt.BatchFlatten(), mlp])

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build

def mnist_conv(activation="relu", # pylint: disable=invalid-name
               batch_norm=True,
               batch_size=128,
               mode="train"):
  """Mnist classification with a multi-layer perceptron."""

  if activation == "sigmoid":
    activation_op = tf.sigmoid
  elif activation == "relu":
    activation_op = tf.nn.relu
  else:
    raise ValueError("{} activation not supported".format(activation))

  # Data.
  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)
  images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
  images = tf.reshape(images, [-1, 28, 28, 1])
  labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")
  if mode == 'train':
      is_training = True
  elif mode == 'test':
      is_training = False
  # Network.
  # mlp = snt.nets.MLP(list(layers) + [10],
  #                    activation=activation_op,
  #                    initializers=_nn_initializers)
  # network = snt.Sequential([snt.BatchFlatten(), mlp])

  def network(inputs, training=is_training):
      # pdb.set_trace()
      def _conv_activation(x):  # pylint: disable=invalid-name
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
              
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs
      if batch_norm:
          linear_activation = lambda x: tf.nn.relu(tf.layers.batch_normalization(x, training=training))
      else:
          linear_activation = tf.nn.relu
      # pdb.set_trace()
      # print(inputs.shape)
      inputs = conv_layer(inputs, 1, 3, 3, 16, "VALID", 'conv_layer1')
      # print(inputs.shape)
      inputs = conv_layer(inputs, 1, 5, 5, 32, "VALID", 'conv_layer2')
      # print(inputs.shape)
      # inputs = linear_activation(inputs)
      inputs = tf.reshape(inputs, [batch_size, -1])
      # print(inputs.shape)
      fc_shape1 = int(inputs.get_shape()[0])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      # print(weights.shape)
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
      # print(bias.shape)
      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

  def build():
    indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
    batch_images = tf.gather(images, indices)
    batch_labels = tf.gather(labels, indices)
    output = network(batch_images)
    return _xent_loss(output, batch_labels)

  return build


CIFAR10_URL = "http://www.cs.toronto.edu/~kriz"
CIFAR10_FILE = "cifar-10-binary.tar.gz"
CIFAR10_FOLDER = "cifar-10-batches-bin"


def _maybe_download_cifar10(path):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(path):
    os.makedirs(path)
  filepath = os.path.join(path, CIFAR10_FILE)
  if not os.path.exists(filepath):
    print("Downloading CIFAR10 dataset to {}".format(filepath))
    url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded {} bytes".format(statinfo.st_size))
    tarfile.open(filepath, "r:gz").extractall(path)


def cifar10(path,  # pylint: disable=invalid-name
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  _maybe_download_cifar10(path)
  # pdb.set_trace()
  # Read images and labels from disk.
  if mode == "train":
    filenames = [os.path.join(path,
                              CIFAR10_FOLDER,
                              "data_batch_{}.bin".format(i))
                 for i in xrange(1, 6)]
    is_training = True
  elif mode == "test":
    filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
    is_training = False
  else:
    raise ValueError("Mode {} not recognised".format(mode))

  depth = 3
  height = 32
  width = 32
  label_bytes = 1
  image_bytes = depth * height * width
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, record = reader.read(tf.train.string_input_producer(filenames))
  record_bytes = tf.decode_raw(record, tf.uint8)

  label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
  image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
  # height x width x depth.
  image = tf.transpose(image, [1, 2, 0])
  image = tf.math.divide(image, 255)

  queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,
                                dtypes=[tf.float32, tf.int32],
                                shapes=[image.get_shape(), label.get_shape()])
  enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

  def network(inputs, training=is_training):
      # pdb.set_trace()
      def _conv_activation(x):  # pylint: disable=invalid-name
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
    
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
            
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs
    
      if batch_norm:
          linear_activation = lambda x: tf.nn.relu(tf.layers.batch_normalization(x, training=training))
      else:
          linear_activation = tf.nn.relu
      # pdb.set_trace()
      # print(inputs.shape)
      inputs = conv_layer(inputs, 2, 3, 3, 16, "VALID", 'conv_layer1')
      # print(inputs.shape)
      inputs = conv_layer(inputs, 2, 5, 5, 32, "VALID", 'conv_layer2')
      # print(inputs.shape)
      # inputs = linear_activation(inputs)
      inputs = tf.reshape(inputs, [batch_size, -1])
      # print(inputs.shape)
      fc_shape1 = int(inputs.get_shape()[0])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      # print(weights.shape)
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
      # print(bias.shape)
      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

  
  def build():
    image_batch, label_batch = queue.dequeue_many(batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])
    # pdb.set_trace()
    output = network(image_batch)
    # print(output.shape)
    return _xent_loss(output, label_batch)

  return build


def vgg16_cifar10(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
    """Cifar10 classification with a convolutional network."""
    
    # Data.
    _maybe_download_cifar10(path)
    # pdb.set_trace()
    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path,
                                  CIFAR10_FOLDER,
                                  "data_batch_{}.bin".format(i))
                     for i in xrange(1, 6)]
        is_training = True
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
        is_training = False
    else:
        raise ValueError("Mode {} not recognised".format(mode))
    
    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)
    
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.math.divide(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    vgg = VGG16(0.5, 10)
    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])
        # pdb.set_trace()
        output = vgg._build_model(image_batch)
        # print(output.shape)
        return _xent_loss(output, label_batch)
    
    return build