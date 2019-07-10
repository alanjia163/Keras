#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin


from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_yaml

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
y_labels = to_categorical(y, num_classes=3)

seed = 7
np.random.seed(seed)


# 构建模型函数
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=3, activation='relu', kernel_initializer=init))

    # 编译模型，
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 构建模型
model = create_model()
model.fit(x, y_labels, epochs=200, batch_size=50, verbose=0)
scores = model.evaluate(x, y_labels, verbose=0)
print(model.metrics_names[1], scores[1] * 100)

model_yaml = model.to_yaml()
with open('./data/model.yaml', 'w') as file:
    file.write(model_yaml)
model.save_weights('./data/model.yaml.h5')

# 加载模型参数
with open('./data/model.yaml', 'r') as file:
    model_read = file.read()

new_model = model_from_yaml(model_read)
new_model.load_weights('./data/model.yaml.h5')

# 编译模型
# 编译模型！！！！
# new_model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
new_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target
y_labels = to_categorical(y, num_classes=3)

scores = new_model.evaluate(x, y_labels, verbose=0)
print(new_model.metrics_names[1], scores[1] * 100)
