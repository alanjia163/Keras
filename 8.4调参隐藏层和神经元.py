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


model = KerasRegressor(build_fn=create_model,epochs=200,batch_size =5,verbose =0)

#调整选择最优模型
param_grid ={}
param_grid['units_list'] = [[20],[13,6]]
param_grid['optimizer'] =['rmsprop','adam']
param_grid['init'] = ['glorot_uniform','normal']
param_grid['epochs'] = [100,200]
param_grid['batch_size'] = [5,20]


#调参
scalar = StandardScaler()
scalar_x = scalar.fit_transform(x)
grid = GridSearchCV(estimator=model,param_grid=param_grid)
results = grid.fit(scalar_x,Y)


#print

print('best:%f  using %s'%(results.best_score_,results.best_params_))
means = results.cv_results_['mean_test_score']
stds = results.cv_results_['std_test_score']
params = results.cv_results_['params']


for mean,std,param in zip(means,stds,params):
    print('%f (%f) with : %r '%(mean,std,param))



