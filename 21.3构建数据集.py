#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
处理过去时间段（t-n ~  t-1）来预测挡墙数据t，这里只需要预测pm2.5浓度，
#风向是分类类型维度，因此需要转换为数字的输入，
'''
from pandas import DataFrame


def convert_dataset(data,n_input=1,out_index=0,dropnan=True):
    n_vars =1 if tpye(data) is list else data.shape[1]
    df =DataFrame(data)
    cols,names=[],[]
    #输入序列（t-n,...,t-1）
    for i in range(n_input,0,-1):
        cols.append(df.shift(i))
        names+=[('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]
        
