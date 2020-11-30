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
Learning 2 Learn evaluation.

这怕不是只是把原来的evaluate里面加了个循环10遍。。。。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
import meta_rnnprop_mt as meta
# import meta
import util
import pdb

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("optimizer", "L2L", "Optimizer.")
flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_integer("num_epochs", 100, "Number of evaluation epochs.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")
flags.DEFINE_integer("bs", 1280, "batch size")
flags.DEFINE_integer("m", 5, "m")
flags.DEFINE_integer("n", 10, "n")
flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("num_steps", 100,
                     "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_string("init", "normal", ".")
flags.DEFINE_float("beta1", 0.95, "")
flags.DEFINE_float("beta2", 0.95, "")
flags.DEFINE_integer("num_mt", 1, "")

def main(_):
  # Configuration.
  num_unrolls = FLAGS.num_steps

  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  # Problem.
  # problem, net_config, net_assignments = util.get_config(FLAGS.problem,
  #                                                        FLAGS.path)
  param_dict = {}
  param_dict['bs'] = FLAGS.bs
  param_dict['m'] = FLAGS.m
  param_dict['n'] = FLAGS.n
  print(param_dict)
  problem, net_config, net_assignments = util.get_config(FLAGS.problem, net_name="RNNprop", 
                                                        num_linear_heads=1, init=FLAGS.init, 
                                                        path=FLAGS.path, param=param_dict)

  # Optimizer setup.
  if FLAGS.optimizer == "Adam":
    cost_op = problem()
    problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    problem_reset = tf.variables_initializer(problem_vars)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
    update = optimizer.minimize(cost_op)
    reset = [problem_reset, optimizer_reset]
  elif FLAGS.optimizer == "L2L":
    if FLAGS.path is None:
      logging.warning("Evaluating untrained L2L optimizer")
    optimizer = meta.MetaOptimizer(FLAGS.num_mt, FLAGS.beta1, FLAGS.beta2, **net_config)
    # meta_loss = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
    
    meta_loss, scale, var_x, constants, subsets, seq_step, \
    loss_mt, update_mt, reset_mt, mt_labels, mt_inputs, hess_norm_approx = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)

    _, update, reset, cost_op, _ = meta_loss

  else:
    raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  final_loss = []
  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<就是在这里加的循环10遍！<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  for i in range(10):
    with ms.MonitoredSession() as sess:
    # with tf.Session(config=config) as sess:
      sess.run(reset)
      # Prevent accidental changes to the graph.
      tf.get_default_graph().finalize()
      total_time = 0
      total_cost = 0
      loss_record = []
      for ep in xrange(FLAGS.num_epochs):
        # Training.
        time, cost = util.run_eval_epoch(sess, cost_op, [update], num_unrolls, step=seq_step)                   
        total_time += time

        total_cost += sum(cost)/num_unrolls
        loss_record += cost
        print(ep, cost[-1])
      # Results.
      util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                      total_time, FLAGS.num_epochs)
    final_loss.append(loss_record)

  print("FINAL LOSS SHAPE:[{}/{}]".format(len(final_loss), len(final_loss[0])))
  with open('{}/{}_eval_loss_record.pickle'.format(FLAGS.path, FLAGS.optimizer), 'wb') as l_record:
    pickle.dump(final_loss, l_record)
  print("Saving evaluate loss record {}".format(FLAGS.path))

if __name__ == "__main__":
  tf.app.run()
