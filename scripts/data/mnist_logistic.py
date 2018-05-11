#!/usr/bin/env python
# coding: utf-8
import time
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image
import numpy as np
import math
import pickle
import os
import torch
import torch.nn as nn
dtype = torch.cuda.FloatTensor
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    def forward(self, x):
        out = self.linear(x)
        return out

def to_np(x):
    return x.data.cpu().numpy()

def main(args):
    input_size = 28*28
    num_classes = 2
    learning_rate = 0.1
    num_epochs = 10

    for count in range(10):
        data = np.load('mnist_1vs7_corrupt_{}.npz'.format(count))
        x_train = data['x_train']
        x_test = data['x_test']
        y_train = data['y_train']
        y_test = data['y_test']
        N = x_train.shape[0]
        T = x_test.shape[0]
        print('done loading')
        model = LogisticRegression(input_size, num_classes)
        # Loss and Optimizer
        # Softmax is internally computed.
        # Set parameters to be updated.
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Training the Model
        batch_size = 100
        print(N)
        print(np.max(y_train))
        y_train[np.where(y_train<0)]=0
        y_test[np.where(y_test<0)]=0
        for epoch in range(num_epochs):
            print(epoch)
            for i in range(int(N/batch_size)):
                images = torch.from_numpy(x_train[i*batch_size:(i+1)*batch_size,:])
                labels = torch.from_numpy(y_train[i*batch_size:(i+1)*batch_size])
                images = Variable(images.view(-1, 28*28))
                labels = Variable(labels)
                #Forward + Backward + Optimize
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if (i+1) % 2 == 0:
                    print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'% (epoch+1, num_epochs, i+1, (N)//batch_size, loss.data[0]))
        correct = 0
        total = 0
        for i in range(int(T/batch_size)):
            images = torch.from_numpy(x_test[i*batch_size:(i+1)*batch_size,:])
            labels = torch.from_numpy(y_test[i*batch_size:(i+1)*batch_size])
            images = Variable(images.view(-1, 28*28))
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        a = model.state_dict()['linear.weight']
        weight = a.numpy()
        b = model.state_dict()['linear.bias']
        bias = b.numpy()
        print(weight.shape)
        print(bias.shape)
        np.save('logistic_weight_{}'.format(count),np.concatenate([weight,np.expand_dims(bias,1)],1))
        print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmbd', type=float, default=0.08)
    parser.add_argument('--batchsize', type=int, default=1000)
    parser.add_argument('--epoch', type=int, default=30000)
    args = parser.parse_args()
    print(args)
    main(args)
