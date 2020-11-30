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
# limitations under the License.
# ==============================================================================
"""
Learning 2 Learn utils.

在原来l2o-dm的代码的基础上加入了rastirgin函数对应代码，要使用必须要在代码文件夹中加入Rastrigin_data
文件夹以提供数据。

三个重要参数：param['bs']=batch_size=1280，param['n']=num_dims_n=2或10，mode='train'或'test'。(param['m']这里不用)

这里batch_size必须在train和test都取1280，因为在test中为了简化代码，直接把test原数据复制10遍放在tensor中，这样执行一次
相当于就对每组数据做了10次test，于是batch_size在这里就必须为1280。

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import random
import pdb


def run_epoch(sess, cost_op, ops, reset, num_unrolls, stddev=None, num_rd=10,
              scale=None, rd_scale=False, rd_scale_bound=3.0, assign_func=None, var_x=None,
              step=None, unroll_len=None,
              task_i=-1, data=None, label_pl=None, input_pl=None, if_hess_init=False, hess_norm=None):
  """Runs one optimization epoch."""
  start = timer()
  # hessian initialization
  if if_hess_init:
      assert hess_norm is not None
      for i in range(100):
          sess.run(reset)
          hn = sess.run(hess_norm)
          prob = np.exp(-0.04/hn)
          if random.random() <= prob:
              break
      # print ("hess{}={}".format(i, hn))
  else:
      sess.run(reset)

  if task_i == -1:
      if rd_scale:
        assert scale is not None
        r_scale = []
        for k in scale:
          r_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                            size=k.shape)))
        assert var_x is not None
        k_value_list = []
        for k_id in range(len(var_x)):
          k_value = sess.run(var_x[k_id])
          k_value = k_value / r_scale[k_id]
          k_value_list.append(k_value)
        assert assign_func is not None
        assign_func(k_value_list)
        feed_rs = {p: v for p, v in zip(scale, r_scale)}
      else:
        feed_rs = {}
      if stddev is not None:
          unroll_len = stddev.get_shape().as_list()[0]
          assert unroll_len >= num_rd
          feed_dict = {**feed_rs, stddev: np.array([0.1]*num_rd+[0.0]*(unroll_len-num_rd))}
          if step is not None:
              feed_dict[step] = 1
          pdb.set_trace()
          cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
          feed_dict = feed_rs
          for i in xrange(1, num_unrolls):
            if step is not None:
                feed_dict[step] = i*unroll_len+1
            cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
      else:
          feed_dict = feed_rs
          for i in xrange(num_unrolls):
            if step is not None:
                feed_dict[step] = i*unroll_len+1
            cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
  else:
      assert data is not None
      assert input_pl is not None
      assert label_pl is not None
      feed_dict = {}
      for ri in xrange(num_unrolls):
          for pl, dat in zip(label_pl, data["labels"][ri]):
              feed_dict[pl] = dat
          for pl, dat in zip(input_pl, data["inputs"][ri]):
              feed_dict[pl] = dat
          if step is not None:
              feed_dict[step] = ri * unroll_len + 1
          print(feed_dict)
          cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
  return timer() - start, cost



# def run_eval_epoch(sess, cost_op, ops, reset, num_unrolls):
#   """Runs one optimization epoch."""
#   start = timer()
#   # sess.run(reset)
#   for _ in xrange(num_unrolls):
#     cost = sess.run([cost_op] + ops)[0]
#   return timer() - start, cost

def run_eval_epoch(sess, cost_op, ops, num_unrolls, step=None):
  """Runs one optimization epoch."""
  start = timer()
  # sess.run(reset)
  total_cost = []
  feed_dict = {}
  for i in xrange(num_unrolls):
    feed_dict[step] = i + 1
    cost = sess.run([cost_op] + ops, feed_dict=feed_dict)[0]
    total_cost.append(cost)
  return timer() - start, total_cost

def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def get_net_path(name, path, idx=0):
  return None if path is None else os.path.join(path, name + ".l2l-{}".format(idx))


def get_default_net_config(name, path, idx=0):
  return {
      "net": "CoordinateWiseDeepLSTM",
      "net_options": {
          "layers": (20, 20),
          "preprocess_name": "LogAndSign",
          "preprocess_options": {"k": 5},
          "scale": 0.01,
      },
      "net_path": get_net_path(name, path, idx)
  }


def get_config(problem_name, param=None, path=None, model_idx=0, mode=None, num_hidden_layer=None, net_name=None, num_linear_heads=1, init="normal"):
  """Returns problem configuration."""
  if problem_name == "simple":
    problem = problems.simple()
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (), "initializer": "zeros"},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  elif problem_name == "simple-multi":
    problem = problems.simple_multi_optimizer()
    net_config = {
        "cw": {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {"layers": (), "initializer": "zeros"},
            "net_path": get_net_path("cw", path)
        },
        "adam": {
            "net": "Adam",
            "net_options": {"learning_rate": 0.01}
        }
    }
    net_assignments = [("cw", ["x_0"]), ("adam", ["x_1"])]
  elif problem_name == "quadratic":
    problem = problems.quadratic(batch_size=128, num_dims=10)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None
  
  elif problem_name == "lasso":
    batch_size = param['bs']
    num_dims_m = param['m']
    num_dims_n = param['n']
    problem = problems.lasso(batch_size=batch_size, num_dims_m=num_dims_m, num_dims_n=num_dims_n)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None

  elif problem_name == "rastrigin":
    batch_size = param['bs']
    num_dims_n = param['n']
    problem = problems.rastrigin(batch_size=batch_size, num_dims=num_dims_n, mode=mode)
    net_config = {"cw": {
        "net": "CoordinateWiseDeepLSTM",
        "net_options": {"layers": (20, 20)},
        "net_path": get_net_path("cw", path)
    }}
    net_assignments = None

  elif problem_name == "mnist":
    if mode is None:
        mode = "train" if path is None else "test"
    problem = problems.mnist(layers=(20,), activation="sigmoid", mode=mode, init="normal")
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None
  elif problem_name == "mnist_relu":
      mode = "test"
      problem = problems.mnist(layers=(20,), activation="relu", mode=mode)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "mnist_deeper":
      mode = "test"
      assert num_hidden_layer is not None
      problem = problems.mnist(layers=(20,) * num_hidden_layer, activation="sigmoid", mode=mode)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "mnist_conv":
      mode = "test"
      problem = problems.mnist_conv(mode=mode, batch_norm=True)
      net_config = {"cw": get_default_net_config("cw", path, model_idx)}
      net_assignments = None
  elif problem_name == "cifar":
    mode = "test"
    problem = problems.cifar10("cifar10", mode=mode)
    net_config = {"cw": get_default_net_config("cw", path, model_idx)}
    net_assignments = None

  elif problem_name == "vgg16":
    mode = "train" if path is None else "test"
    problem = problems.vgg16_cifar10("cifar10",
                               mode=mode)
    net_config = {"cw": get_default_net_config("cw", path)}
    net_assignments = None
  elif problem_name == "cifar-multi":
    mode = "train" if path is None else "test"
    problem = problems.cifar10("cifar10",
                               conv_channels=(16, 16, 16),
                               linear_layers=(32,),
                               mode=mode)
    net_config = {
        "conv": get_default_net_config("conv", path),
        "fc": get_default_net_config("fc", path)
    }
    conv_vars = ["conv_net_2d/conv_2d_{}/w".format(i) for i in xrange(3)]
    fc_vars = ["conv_net_2d/conv_2d_{}/b".format(i) for i in xrange(3)]
    fc_vars += ["conv_net_2d/batch_norm_{}/beta".format(i) for i in xrange(3)]
    fc_vars += ["mlp/linear_{}/w".format(i) for i in xrange(2)]
    fc_vars += ["mlp/linear_{}/b".format(i) for i in xrange(2)]
    fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  if net_name == "RNNprop":
      default_config = {
              "net": "RNNprop",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "fc",
                  "preprocess_options": {"dim": 20},
                  "scale": 0.01,
                  "tanh_output": True,
                  "num_linear_heads": num_linear_heads
              },
              "net_path": get_net_path("rp", path, model_idx)
          }
      net_config = {"rp": default_config}

  return problem, net_config, net_assignments
