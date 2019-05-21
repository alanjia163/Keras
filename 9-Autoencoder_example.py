#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np

from matplotlib import pyplot as plt

np.random.seed(10)
from keras.models import Model  # 此处不用Sequencial
from keras.layers import Dense, Input
from keras.datasets import mnist

# data
(x_train, _), (x_test, y) = mnist.load_data()
# x_y_pre_procession
x_train = x_train.astype('float32') / 255. - 0.5  # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5  # minmax_normalized

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)

# in order to plot in a 2D figure
encoding_dim = 2

# input
input_img = Input(shape=(784,))

# layers
# encoders
el1 = Dense(128, activation='relu')(input_img)
el2 = Dense(64, activation='relu')(el1)
el3 = Dense(32, activation='relu')(el2)
encoder_out = Dense(encoding_dim)(el3)
# decoders
dl1 = Dense(64, activation='relu')(encoder_out)
dl2 = Dense(128, activation='relu')(dl1)
dl3 = Dense(128, activation='relu')(dl2)
decoder_out = Dense(784, activation='relu')(dl3)

#construct the autoencoder model
autoencoder = Model(input=input_img, output=decoder_out)

#
encoder = Model(input = input_img,output = encoder_out)

autoencoder.compile(optimizer='adam',loss = 'mse')

#training
autoencoder.fit(x_train,x_train,batch_size=100,epochs=20,shuffle=True)

#预测
encoded_imgs = encoder.predict(x_test)
# plotting
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y)
plt.colorbar()
plt.show()


