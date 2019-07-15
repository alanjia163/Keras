#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin


from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# data
dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

y_labels = to_categorical(y, num_classes=3)
seed = 2019
np.random.seed(seed)


# 构建模型
def create_model(optimizer='rmsprop', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4, activation='relu', input_dim=4, kernel_initializer=init))
    model.add(Dense(units=6, activation='relu', kernel_initializer=init))
    model.add(Dense(units=6, activation='relu',  kernel_initializer=init))
    model.add(Dense(units=6, activation='sigmoid', kernel_initializer=init))
    model.add(Dense(units=3, activation='softmax', kernel_initializer=init))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

#model
model = create_model()
#checkpoint
filepath = 'weight.best.h5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_best_only=True,mode ='max')
callback_list = [checkpoint]
model.fit(x,y_labels,validation_split=0.2,epochs=200,batch_size=5,verbose=0,callbacks=callback_list)


