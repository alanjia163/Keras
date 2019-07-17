#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
'''
徐贯模型把网络表示层网络层的管道或列表，
'''
from keras import Model, Input
from keras.layers import TimeDistributed
from keras.models import  Sequential
from keras.layers.core import Dense,Activation
model1 =Sequential([
    Dense(32,input_dim=784),
    Activation('sigmoid'),
    Dense(10,),
    Activation('softmax'),
    ]
)


#输入-输出型网络
inputs =Input(shape=(784,))
x =Dense(32)(inputs)
x =Activation('sigmoid')(x)
x =Dense(10)(x)
predictions =Activation('softmax')(x)
model2 = Model(inputs=inputs,outputs=predictions)
model2.compile(loss='categorical_crossentropy',optimizer='adam')


#包装模型扩展
sequence_prediction = TimeDistributed(model1)(input_sequences)


#多输入输出
model = Model(inputs=[input1,input2],outputs=[output1,output2])




