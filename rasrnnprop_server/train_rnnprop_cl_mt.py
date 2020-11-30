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
Learning 2 Learn training.

加入了mode以将rastrigin的train与test区别开，由于本程序（train_rnnprop_cl_mt.py）本来就是一个train程序，
于是这里默认是train，不必额外指明。
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
import pickle
import meta_rnnprop_mt as meta
import util
import os
from data_generator import data_loader
import random
import pdb
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

flags = tf.flags
logging = tf.logging


FLAGS = flags.FLAGS
flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")
flags.DEFINE_string("log_path", None, "Path for saved logs.")
flags.DEFINE_integer("num_epochs", 100000, "Number of training epochs.")
flags.DEFINE_integer("log_period", 1, "Log period.")
flags.DEFINE_integer("evaluation_period", 10, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 20, "Number of evaluation epochs.")

flags.DEFINE_integer("bs", 1280, "batch size")
flags.DEFINE_integer("m", 5, "m")
flags.DEFINE_integer("n", 10, "n")
flags.DEFINE_string("mode",'train',"Mode(train or test).Should be train in this program.")
#这里加入mode以将rastrigin的train与test区别开，由于本程序（train_rnnprop_cl_mt.py）本来就是一个train程序，
#于是这里默认是train，不必额外指明。

flags.DEFINE_string("problem", "simple", "Type of problem.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")

flags.DEFINE_float("rd_scale_bound", 3.0, "Bound for random scaling on the main optimizee.")
flags.DEFINE_float("beta1", 0.95, "")
flags.DEFINE_float("beta2", 0.95, "")
flags.DEFINE_integer("num_mt", 1, "")
flags.DEFINE_string("optimizers", "adam", ".")
flags.DEFINE_float("mt_ratio", 0.3, "")
flags.DEFINE_boolean("if_scale", True, "Use second derivatives.")
flags.DEFINE_integer("k", 1, "")
flags.DEFINE_string("init", "normal", ".")

def main(_):
  # Configuration.
  num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]       # steps number
  num_unrolls = [int(ns/FLAGS.unroll_length) for ns in num_steps] # unroll number
  num_unrolls_eval = num_unrolls[1:]
  min_num_eval = 3
  curriculum_idx = 0

  if FLAGS.save_path is not None:
    if not os.path.exists(FLAGS.save_path):
      os.mkdir(FLAGS.save_path)
  if FLAGS.log_path is not None:
      if not os.path.exists(FLAGS.log_path):
          os.mkdir(FLAGS.log_path)

  # Problem.
  param_dict = {}
  param_dict['bs'] = FLAGS.bs
  param_dict['m'] = FLAGS.m
  param_dict['n'] = FLAGS.n
  print(param_dict)


  problem, net_config, net_assignments = util.get_config(FLAGS.problem, net_name="RNNprop", mode=FLAGS.mode,#加入mode
                                         num_linear_heads=1, init=FLAGS.init, param=param_dict)
  
  
  # Optimizer setup.
  optimizer = meta.MetaOptimizer(FLAGS.num_mt, FLAGS.beta1, FLAGS.beta2, **net_config)
  minimize, scale, var_x, constants, subsets, seq_step, \
  loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, hess_norm_approx = optimizer.meta_minimize(
      problem, FLAGS.unroll_length,
      learning_rate=FLAGS.learning_rate,
      net_assignments=net_assignments,
      second_derivatives=FLAGS.second_derivatives)

  step, update, reset, cost_op, _ = minimize
  # pdb.set_trace()
  data_mt = data_loader(problem, var_x, constants, subsets, scale,
                        FLAGS.optimizers, FLAGS.unroll_length, None)

  p_val_x = []
  for k in var_x:
      p_val_x.append(tf.placeholder(tf.float32, shape=k.shape))
  assign_ops = [tf.assign(var_x[k_id], p_val_x[k_id]) for k_id in range(len(p_val_x))]

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  with ms.MonitoredSession() as sess:
    def assign_func(val_x):
      sess.run(assign_ops, feed_dict={p: v for p, v in zip(p_val_x, val_x)})

    tf.get_default_graph().finalize()

    best_evaluation = 10000000000
    loss_record = []
    task_id_record = []
    train_loss_record = []
    eval_loss_record = []
    num_steps_train = []
    num_eval = 0
    improved = False
    mti = -1
    
    for e in xrange(FLAGS.num_epochs):
      # change task
      if random.random() < FLAGS.mt_ratio:
        mti = (mti+1)%FLAGS.num_mt
        task_i = mti
      else:
        task_i = -1
      task_id_record.append(task_i)
      # Training.
      if task_i == -1:

          time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                      num_unrolls[curriculum_idx],
                                      scale=scale,
                                      rd_scale=FLAGS.if_scale,
                                      rd_scale_bound=FLAGS.rd_scale_bound,
                                      assign_func=assign_func,
                                      var_x=var_x,
                                      step=seq_step,
                                      unroll_len=FLAGS.unroll_length,
                                      if_hess_init=FLAGS.init == "hessian",
                                      hess_norm=hess_norm_approx)
          
      else:
          data_e = data_mt.get_data(task_i, sess, num_unrolls[curriculum_idx], assign_func, FLAGS.rd_scale_bound, if_scale=FLAGS.if_scale, mt_k=FLAGS.k, if_hess_init=FLAGS.init=="hessian")
          time, cost = util.run_epoch(sess, loss_mt[task_i], [update_mt[task_i], steps_mt[task_i]], reset_mt[task_i],
                                      num_unrolls[curriculum_idx],
                                      scale=scale,
                                      rd_scale=FLAGS.if_scale,
                                      rd_scale_bound=FLAGS.rd_scale_bound,
                                      assign_func=assign_func,
                                      var_x=var_x,
                                      step=seq_step,
                                      unroll_len=FLAGS.unroll_length,
                                      task_i=task_i,
                                      data=data_e,
                                      label_pl=mt_labels[task_i],
                                      input_pl=mt_inputs[task_i])
      train_loss_record.append(cost)

      # Evaluation.
      if (e+1) % FLAGS.evaluation_period == 0:
          num_eval += 1

          eval_cost = 0
          for _ in xrange(FLAGS.evaluation_epochs):
            time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                        num_unrolls_eval[curriculum_idx],
                                        step=seq_step,
                                        unroll_len=FLAGS.unroll_length)
            eval_cost += cost
          print ("epoch={}, num_steps={}, eval loss={}".format(e, num_steps[curriculum_idx], eval_cost/FLAGS.evaluation_epochs), flush=True)
          eval_loss_record.append(eval_cost/FLAGS.evaluation_epochs)
          num_steps_train.append(num_steps[curriculum_idx])

          # update curriculum
          
          if eval_cost < best_evaluation:
            best_evaluation = eval_cost
            improved = True
            # save model
            optimizer.save(sess, FLAGS.save_path, curriculum_idx)
            optimizer.save(sess, FLAGS.save_path, 0)
          elif num_eval >= min_num_eval and improved:
            # restore model
            optimizer.restore(sess, FLAGS.save_path, curriculum_idx)
            num_eval = 0
            improved = False
            curriculum_idx += 1
            if curriculum_idx >= len(num_unrolls):
              print("end of num_unrolls")
              break
            # new evaluation
            eval_cost = 0
            for _ in xrange(FLAGS.evaluation_epochs):
              time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                          num_unrolls_eval[curriculum_idx],
                                          step=seq_step,
                                          unroll_len=FLAGS.unroll_length)
              eval_cost += cost
            best_evaluation = eval_cost
            print("epoch={}, num_steps={}, eval loss={}".format(e, num_steps[curriculum_idx], eval_cost / FLAGS.evaluation_epochs), flush=True)
            eval_loss_record.append(eval_cost / FLAGS.evaluation_epochs)
            num_steps_train.append(num_steps[curriculum_idx])
          elif num_eval >= min_num_eval and not improved:
            print ("no improve during curriculum {} --> stop".format(curriculum_idx))
            break

    # # output
    # print("total time = {}s...".format(timer() - start_time))
    # output
    with open('{}/log.pickle'.format(FLAGS.save_path), 'wb') as l_record:
        records = {"eval_loss": eval_loss_record,
                   "train_loss": train_loss_record,
                   "task_id": task_id_record,
                   "num_steps": num_steps_train}
        pickle.dump(records, l_record)
        l_record.close()


if __name__ == "__main__":
  tf.app.run()
