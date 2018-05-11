from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import sys
sys.path.append('../')
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from tensorflow.contrib.learn.python.learn.datasets import base

import scipy
import sklearn

import influence.experiments as experiments
import influence.dataset as dataset
from influence.dataset import DataSet
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_mnist import load_small_mnist, load_mnist

import tensorflow as tf

np.random.seed(42)

data_sets = load_mnist('data')

# Filter out two classes
pos_class = 1
neg_class = 7

X_train = data_sets.train.x
Y_train = data_sets.train.labels
X_test = data_sets.test.x
Y_test = data_sets.test.labels

X_train, Y_train = dataset.filter_dataset(X_train, Y_train, pos_class, neg_class)
X_test, Y_test = dataset.filter_dataset(X_test, Y_test, pos_class, neg_class)

## If using logistic regression to train
lr_train = DataSet(X_train, np.array((Y_train + 1) / 2, dtype=int))
lr_validation = None
lr_test = DataSet(X_test, np.array((Y_test + 1) / 2, dtype=int))
lr_data_sets = base.Datasets(train=lr_train, validation=lr_validation, test=lr_test)

num_classes = 2
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.01
batch_size = 100
initial_learning_rate = 0.001 
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

num_params = 784

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=lr_data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist-%dvs%d-logreg'%(pos_class, neg_class))

#tf_model.train()
tf_model.load_checkpoint(0)

##############################################
### Flipping experiment
##############################################
X_train = np.copy(tf_model.data_sets.train.x)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels) 

num_train_examples = Y_train.shape[0] 
num_flip_vals = 6
num_check_vals = 6
num_random_seeds = 40

dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)
fixed_influence_loo_results = np.zeros(dims)
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)
fixed_our_results = np.zeros(dims)

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

# Save which indices were flipped to start with
flipped_indices  = dict()

# Save the original result without flipping
orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op], 
    feed_dict=tf_model.all_test_feed_dict)
            
print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

for flips_idx in range(num_flip_vals):
    tmp_flips = dict()
    for random_seed_idx in range(num_random_seeds):
        
        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)        
        np.random.seed(random_seed)
    
        num_flips = int(num_train_examples / 7) * (flips_idx + 1)    
        idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False)
        Y_train_flipped = np.copy(Y_train)
        Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip] 

        # Save the indicies that were flipped for this particular experiment
        tmp_flips[random_seed_idx] = idx_to_flip
        
        tf_model.update_train_x_y(X_train, Y_train_flipped)
        tf_model.train()        
        flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
            [tf_model.loss_no_reg, tf_model.accuracy_op], 
            feed_dict=tf_model.all_test_feed_dict)
        print('Flipped loss: %.5f. Accuracy: %.3f' % (
                flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))
        
        print(flips_idx, num_flips, num_train_examples)

        train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)
        train_loo_influences = tf_model.get_loo_influences()
        # TODO We need our influence here

        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 10) * (checks_idx + 1)

            print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

            fixed_influence_loo_results[flips_idx, checks_idx, random_seed_idx, :], \
              fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :], \
              fixed_random_results[flips_idx, checks_idx, random_seed_idx, :] \
              fixed_ours_results[flips_idx, checks_idx, random_seed_idx, :] \
              = experiments.test_mislabeled_detection_batch(
                tf_model, 
                X_train, Y_train,
                Y_train_flipped,
                X_test, Y_test, 
                train_losses, train_loo_influences, ours_influences
                num_flips, num_checks)

    flipped_indices[flips_idx] = tmp_flips

np.savez(
    'output/mnist%dvs%d_labelfix_results'%(pos_class, neg_class), 
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results,
    fixed_ours_results=fixed_ours_results,
    flipped_indices=flipped_indices
)
