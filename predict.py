#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.models import model_from_json






# 从json文件中加载模型
with open('model.json', 'r') as file:
    model_json = file.read()

new_model = model_from_json(model_json)
new_model.load_weights('first_try.h5')
new_model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# #编译模型！！！！
# #new_model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
# new_model.compile(loss = 'mse',optimizer='rmsprop',metrics=['accuracy'])
#
# scores =new_model.evaluate(x,y_labels,verbose=0)
# print(new_model.metrics_names[1], scores[1] * 100)




# 加载图像

class_img =['cardboard','glass','metal','paper','plastic','trash']
path='./predict/glass1.jpg'


img = load_img(path,target_size=(150, 150))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)
classes = new_model.predict_classes(img)
print(classes)
print(class_img[classes[0]])
#
# class_list =['glass','trash']
# for cl in classes:
#     if cl <=len(class_list):
#         print(class_list[int(cl)])
