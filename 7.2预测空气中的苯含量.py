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
AIRQUALITY_FILE =os.path.join(DATA_DIR,'AirQualityUCI.csv')
aqdf=pd.read_csv(AIRQUALITY_FILE,sep=';',decimal=',',header=0)

#移除前面和后面俩列
del aqdf['Date']
del aqdf['Time']
del aqdf['Unnamed: 15']
del aqdf['Unnamed: 16']

#使用平均值填充空缺值,并将数据导出为矩阵
aqdf=aqdf.fillna(aqdf.mean())
Xorig = aqdf.as_matrix()
# Xorig = aqdf.values


#标准化
scaler = StandardScaler()
Xscaler = scaler.fit_transform(Xorig)
#保存均值和标准差，以用于预测新数据
Xmeans = scaler.mean_
Xstds = scaler.scale_

#数据集，
y=Xscaler[:,3]#目标为第四列
train_size = int(0.7*X.shape)



