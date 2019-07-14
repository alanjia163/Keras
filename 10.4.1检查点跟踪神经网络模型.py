#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


#导入数据
dataset = datasets.load_iris()

x = dataset.data
Y = dataset.target
#将标签转换为分类编码
y_labels = to_categorical(Y,num_classes=3)

#
seed=7
np.random.seed(seed)

#构建模型
def create_model(optimizer = 'rmsprop',init = 'glorot_uniform'):
    model = Sequential()
    model.add(Dense(units=4,activation='relu',input_dim=4,kernel_initializer=init))
    model.add(Dense(units=6,activation='relu',kernel_initializer=init))
    model.add(Dense(units=3,activation='softmax',kernel_initializer=init))

    #compile
    model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return model


#model
model = create_model()

#设置检查点，
filepath = 'weithts-improvement-{epoch:02d}-{val_acc:.2f}.h5'
checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_best_only=True,mode ='max')
calllback_list = [checkpoint]

#fit
model.fit(x,y_labels,validation_data=0.2,epochs=200,batch_size=5,verbose=0,callbacks=calllback_list)


