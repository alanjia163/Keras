#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
对于5中不同的花朵进行分类
'''

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Lambda, Cropping2D
from keras.utils import np_utils
from keras import optimizers

seed = 2019

# data
DATA_DIR = './data'
images = glob.glob(DATA_DIR + 'flower_photos/*/*.jpg')
# 从文件名提取标号
labels = [x.split('/')[2] for x in images]
# 查看数据
unique_labels = set(labels)
plt.figure(figsize=(10, 8))
i = 1
for label in unique_labels:
    image = images[labels.index(label)]
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(5, 5, i)
    plt.title('{0}{1}'.format(label, labels.count(label)))
    i+=1
    _ = plt.imshow(img)
plt.show()



encoder = LabelBinarizer()
encoder.fit(labels)
y = encoder.transform(labels).astype(float)
x_train,x_val,y_train,y_val =train_test_split(images,y,test_size=0.1,random_state=seed)

#model
model = Sequential()
model.add(Lambda(lambda x:(x/255..)-0.5,input_shape=(100,100,3)))
model.add(Conv2D(16,(5,5),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(32,(5,5),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5,activation='softmax'))

#optimizer


