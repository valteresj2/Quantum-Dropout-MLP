# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:34:18 2019

@author: valter.e.junior
"""

import numpy as np
from random import sample
import pickle, gzip
import matplotlib.pyplot as plt
import mlp_drop_quantum as mlp

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    

train_X = valid_set[0]
train_y = valid_set[1]  
print('Shape of training set: ' + str(train_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(train_y)
labels = np.unique(train_y)
train_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(train_y == labels[ix_label])[0]
    train_Y[ix_tmp, ix_label] = 1


# Test data
test_X = test_set[0]
test_y = test_set[1] 
print('Shape of test set: ' + str(test_X.shape))

# change y [1D] to Y [2D] sparse array coding class
n_examples = len(test_y)
labels = np.unique(test_y)
test_Y = np.zeros((n_examples, len(labels)))
for ix_label in range(len(labels)):
    # Find examples with with a Label = lables(ix_label)
    ix_tmp = np.where(test_y == labels[ix_label])[0]
    test_Y[ix_tmp, ix_label] = 1

X=train_X
Y=train_Y
mlp_classifier = mlp.Mlp(size_layers = [784, 25,10, 10], 
                         act_funct   = 'relu',
                         reg_lambda  = 0,
                         bias_flag   = True,dropout=0.2,type_dropout='quantum')
epochs=200
loss=[]
for ix in range(epochs):
    mlp_classifier.train(X, Y, 1)
    Y_hat = mlp_classifier.predict(X)
    # loss
    loss.append((0.5)*np.square(Y_hat - Y).mean())
    print(ix)