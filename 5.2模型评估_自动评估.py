#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
通过fit()函数分割参数,设置数据集百分比,
'''

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(100)

# data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')  # 总共有9维数据,最后一个维度为0-1标签

x = dataset[:, 0:8]
y = dataset[:, 8:]

# model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# compile
model.compile(
    loss='binary_corssentropy', optimizer='adam',
    metrics=['accuracy'],
)

# 训练模型并评估
model.fit(x=x,y=y,epochs=150, batch_size=20,validation_split=0.2)

