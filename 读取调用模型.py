#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json

# 从json文件中加载模型
with open('./data/model.json', 'r') as file:
    model_json = file.read()

new_model = model_from_json(model_json)
new_model.load_weights('./data/model.json.h5')

#编译模型！！！！
#new_model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
new_model.compile(loss = 'mse',optimizer='rmsprop',metrics=['accuracy'])

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
y_labels = to_categorical(y, num_classes=3)

scores =new_model.evaluate(x,y_labels,verbose=0)
print(new_model.metrics_names[1], scores[1] * 100)
