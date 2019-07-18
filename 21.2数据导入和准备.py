#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jia ShiLin
'''

'./data/pollution_origianl.csv'
Attribute Information:

No: row number
year: year of data in this row
month: month of data in this row
day: day of data in this row
hour: hour of data in this row
pm2.5: PM2.5 concentration (ug/m^3)
DEWP: Dew Point (â„ƒ)
TEMP: Temperature (â„ƒ)
PRES: Pressure (hPa)
cbwd: Combined wind direction
Iws: Cumulated wind speed (m/s)
Is: Cumulated hours of snow
Ir: Cumulated hours of rain

'''



from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
file ='./data/pollution_original.csv'

def prase(x):
    return datetime.strptime(x,'%Y %m %d %H')

def load_dataset():
    #
    dataset = read_csv(file,parse_dates=[['year','month','day','hour']],index_col=0,date_parser=prase)

    #删除No列
    dataset.drop('No',axis=1,inplace=True)

    #设定列名
    dataset.columns = ['pollution','dew','temp','press','wnd_dir','wnd_spd','snow','rain']
    dataset.index.name='date'

    #使用中位数填充缺失值
    dataset['pollution'].fillna(dataset['pollution'].mean(),inplace=True)

    return dataset

if __name__ == '__main__':
    dataset =load_dataset()
    print(dataset.head(5))
    groups =[0,1,2,3,5,6,7]
    plt.figure(figsize=(8,5))

    i=1
    for group in groups:
        plt.subplot(len(groups),1,i)
        plt.plot(dataset.values[:,group])
        plt.title(dataset.columns[group],y=0.5,loc='right')
        i=i+1

    plt.show()

