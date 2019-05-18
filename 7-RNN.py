#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np

np.random.seed(10)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, activations, SimpleRNN
from keras.optimizers import Adam

# hyper parameter
TIME_STEPS = 28
INPUT_SIZE = 28
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR =0.01

#data
(input_train,output_train),(input_test,output_test) = mnist.load_data()
#pre_processiong
input_train = input_train.reshape(-1,28,28) / 255
input_test = input_test.reshape(-1,28,28) / 255

output_train =np_utils.to_categorical(output_train,num_classes=10)
output_test =np_utils.to_categorical(output_test,num_classes=10)

model = Sequential()

model.add(SimpleRNN(
    batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
    output_dim = CELL_SIZE,
    unroll=True,
))

#optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss = 'categorical_crossentropy',
              metrics=['accuracy']
              )
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = input_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
    Y_batch = output_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
    if step % 500 == 0:
        cost, accuracy = model.evaluate(input_test, output_test, batch_size=output_test.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)

