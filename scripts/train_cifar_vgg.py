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
from influence.cifar_mlp import CIFAR_MLP
import pickle

from load_mnist import load_small_mnist, load_mnist
from load_cifar import *
from gen_vgg_features import *
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
import h5py

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# First create the dataset object from the VGG features
print("Loading Data...")
#with open('output_31_train_features.pkl', 'rb') as f:
#    x_train = pickle.load(f)
#with open('output_31_test_features.pkl', 'rb') as f:
#    x_test = pickle.load(f)

hf = h5py.File('data/vgg_features_cifar.h5', 'r')
x_train = np.array(hf.get('train'))
x_test = np.array(hf.get('test'))
hf.close()

y_train, y_test = load_cifar_labels()
train = DataSet(x_train, y_train.flatten())
test = DataSet(x_test, y_test.flatten())
data_sets = base.Datasets(train=train, validation=None, test=test)

num_classes = 10
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.00001 
decay_epochs = [10000, 20000]

model = CIFAR_MLP(
    input_dim=25088,
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
    model_name='cifar_mlp')

num_steps = 500000
model.train(
    num_steps=num_steps, 
    iter_to_switch_to_batch=10000000,
    iter_to_switch_to_sgd=10000000)
iter_to_load = num_steps - 1

print('Training done')

"""
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

print('Training done')
"""
# Load the trained model
iter_to_load = 99999
model.load_checkpoint(iter_to_load)

# compute influence values for the set of test points
test_indices = [6]

num_train = len(model.data_sets.train.labels)
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

np.savez('output/cifar_inf_test_mlp%s'%(test_indices), influences=influences)
