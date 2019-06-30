#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from sklearn import datasets
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import  KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from  sklearn.model_selection import GridSearchCV


#data 使用的是sklearn中的数据集
dataset = datasets.load_boston()

x = dataset.data
Y = dataset.target

#
seed = 7
np.random.seed(seed）


#模型函数
def create_model(units_list = [13],optimizer = 'adam',init = 'normal'):
    #构建模型
    model = Sequential()
    #add layers
    model.add(Dense(units=units_list[0],activation='relu',input_dim=13,kernel_initializer=init))
    for units  in units_list[1:]:
        model.add(Dense(units=units,activation='relu',kernel_initializer=init,kernel_initializer=init))

    model.compile(loss='mean_squared_error',optimizer=optimizer)

    return model

#回归模型包装类KerasRegressor,将必要参数传递给模型的fit()函数，如epochs,和batch_size
model = KerasRegressor(build_fn=create_model,epochs=200,batch_size =5,verbose =0)

steps = []
steps.append(('standardize',StandardScaler()))
steps.append(('mlp',model))
pipeline=Pipeline(steps)

#10折交叉验证评估模型,
kfold = KFold(n_splits=10,shuffle=True,random_state=seed)
results = cross_val_score(pipeline,x,Y,cv=kfold)

print('Baseline:%.2f(%.2f)MSE'%(results.mean(),results.std()))


