from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np

import tensorflow as tf

import os, sys
sys.path.append('../')

import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C
from influence.cifar_cnn import CIFAR_CNN

from load_mnist import load_small_mnist, load_mnist
from load_cifar import *
from gen_vgg_features import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

data_sets = generate_vgg_features()

num_classes = 10
input_side = 224
input_channels = 3
input_dim = input_side * input_side * input_channels 
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001 
decay_epochs = [10000, 20000]
hidden1_units = 128
hidden2_units = 128
hidden3_units = 128
conv_patch_size = 3
keep_probs = [1.0, 1.0]


model = CIFAR_CNN(
    input_side=input_side, 
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units, 
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes, 
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output', 
    log_dir='log',
    model_name='cifar_all_cnn_c')

num_steps = 500000
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1

test_idx = 6

actual_loss_diffs, predicted_loss_diffs, indices_to_remove = experiments.test_retraining(
    model, 
    test_idx=test_idx, 
    iter_to_load=iter_to_load, 
    num_to_remove=100,
    num_steps=30000, 
    remove_type='maxinf',
    force_refresh=True)

np.savez(
    'output/cifar_all_cnn_c_iter-500k_retraining-100.npz', 
    actual_loss_diffs=actual_loss_diffs, 
    predicted_loss_diffs=predicted_loss_diffs, 
    indices_to_remove=indices_to_remove
    )

# Load the trained model
model.load_checkpoint(499999)

# compute influence values for the set of test points
test_indices = [6]

num_train = len(.data_sets.train.labels)
influences = None

for test_idx in test_indices:
    influence = model.get_influence_on_test_loss(
                 [test_idx],
                 np.arange(num_train),
                 force_refresh=True)
    influence = np.transpose(np.array([influence]))
    if influences is not None:
        influences = np.append(influences, influence, 1)
    else:
        influences = influence                                                                                                                                                                                                

np.savez('output/cifar_inf_test%s'%(test_indices), influences=influences)
