#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
'''
数据集是UCI库中的

读入用pandas，CSV文件，原因1.包含空字段，2.数据集为欧洲国家通常习惯的逗号作为小数点
'''

from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

DATA_DIR = './data'
AIRQUALITY_FILE = os.path.join(DATA_DIR, 'AirQualityUCI.csv')
aqdf = pd.read_csv(AIRQUALITY_FILE, sep=';', decimal=',', header=0)

# 移除前面和后面俩列
del aqdf['Date']
del aqdf['Time']
del aqdf['Unnamed: 15']
del aqdf['Unnamed: 16']

# 使用平均值填充空缺值,并将数据导出为矩阵
aqdf = aqdf.fillna(aqdf.mean())
Xorig = aqdf.as_matrix()
# Xorig = aqdf.values


# 标准化
scaler = StandardScaler()
Xscaler = scaler.fit_transform(Xorig)
# 保存均值和标准差，以用于预测新数据
Xmeans = scaler.mean_
Xstds = scaler.scale_

# 数据集，
x = np.delete(Xscaler, 3, axis=1)
y = Xscaler[:, 3]  # 目标为第四列

train_size = int(0.7 * X.shape[0])  # 划分数据集的长度
xtrain, xtest, ytrain, ytest = x[0:train_size], x[train_size:], y[:train_size], y[train_size:]

# layers,输入是一个12特征的向量，输出是伸缩值预测，hiddenlayers有8个神经元
# 初始化机制glorot uniform来初始化全连接权重矩阵，
readings = Input(shape=(12,))
x = Dense(8, activation='relu',kernel_initializer = 'glorot_uniform')(readings)
benzene = Dense(1,kernel_initializer='glorot_uniform')(x)
model =Model(inpus =[readings],outputs =[benzene])
model.compile(loss='mse',optimizer='adam')


#batch,epoch
NUM_EPOCHS =20
BATCH_SIZE =10

history =model.fit(xtrain,ytrain,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_split=0.2)
ytest_ = model.predict(xtest).flatten()

#预测值和真实值对比，将伸缩后的z值重新伸缩
for i in range(10):
    label =(ytest[i]*Xstds[3])+Xmeans[3]
    prediction =(ytest_[i]*Xstds[3])+Xmeans[3]
    print('Benzene Conc. expected:{:.3f},prdicted:{:.3f}'.format(label,prediction))

#最后绘制
plt.plot(np.arange(ytest.shape[0]),(ytest*Xstds[3])/Xmeans[3],color='b',label='actual')
plt.plot(np.arange(ytest_.shape[0]),(ytest_*Xstds[3])/Xmeans[3],color='r',alpha=0.5,label='predicted')
plt.xlabel('time')
plt.ylabel('苯含量')
plt.legend(loc ='best')
plt.show()


