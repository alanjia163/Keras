#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold

import numpy as np

from sklearn import datasets

# data
dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

#
seed = 7
np.random.seed(seed)


# 构建模型参数
def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))

    model.add(Dense(units=4, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='relu', kernel_initializer=init))

    # compile
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=2000, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)

print('accuracy:%.2f%%(%.2f)' % (results.mean() * 100, results.std()))

# accuracy: 96.67%(0.04)
