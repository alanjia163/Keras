#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier


# 构建模型
def create_model():
    model = Sequential()
    model.add(Dense(units=12, input_dim=8, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy', )

    return model


seed = 7
np.random.seed(seed)

# data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# x,y
x = dataset[:, 0:8]
y = dataset[:, 8:]

# 创建模型 for scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# 10折交叉验证
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, x, y, cv=kfold)
print(results.mean())
# 输出结果:0.65934792372923
