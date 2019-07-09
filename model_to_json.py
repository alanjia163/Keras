#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
#模型的权重保存在HDF5中
#模型的结构保存在JSON文件或者YAML文件中
'''

'''
#Keras提供了to_json()生成模型JSON描述，并将模型的JSON描述保存到文件中，
#反序列化时候，通过model_from_json()函数加载模型描述，编译生成模型

##save_weights()函数可以保存模型的权重值，加载时使用load_weights()
##eg:训练一个模型，使用json格式描述模型结构，保存model.json文件中，权重信息保存到本地目录model.json.h5文件中

###当新的数据需要预测时候，加载模型和权重信息，创建新模型，必须先编译，

'''

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json

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

# 保存模型
model_json = model.to_json()
with open('./data/model.json', 'w') as file:
    file.write(model_json)

# 保存模型权重值
model.save_weights('./data/model.json.h5')

