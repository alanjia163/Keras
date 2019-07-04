#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin


import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D

# 指定数据目录，提取类别全部模式文件名,此数据集中照片的维度不同
DATA_DIR = 'Data/PetImages'
cats = glob.glob(DATA_DIR + 'Cat/*.jpg')
dogs = glob.glob(DATA_DIR + 'Dog/*.jpg')

print('cats:%d' % len(cats))
print('dogs:%d' % len(dogs))
##cats:12500
##dogs:12500

SEED = 2019

# 随机绘制每个类别中三个图
n_examplts = 3
plt.figure(figsize=(5, 5))
i = 1
for _ in range(n_examplts):
    image_cat = cats[np.random.randint(len(cats))]
    img_cat = cv2.imread(image_cat)
    img_cat = cv2.cvtColor(img_cat, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 2, i)
    _ = plt.imshow(img_cat)
    i += 1

    image_dog = dogs[np.random.randint(len(dogs))]
    img_dog = cv2.imread(image_dog)
    img_dog = cv2.cvtColor(img_dog, cv2.COLOR_BGR2RGB)
    plt.subplot(3, 2, i)
    i += 1
    _ = plt.imshow(img_dog)

plt.show()

# 数据集划分
dogs_train, dogs_val, cats_train, cats_val = train_test_split(dogs, cats, test_size=0.2, random_state=SEED)


# 训练集较大，使用批生成器，
def batchgen(cats, dogs, batch_size, img_size=50):
    # 创建空Numpy数组
    batch_images = np.zeros((batch_size, img_size, img_size, 3))
    batch_label = np.zeros(batch_size)

    # 创建批量样本生成器
    while 1:
        n = 0
        while n < batch_size:
            # 随机挑选一张狗或者猫的图像
            if np.random.randint(2) == 1:
                i = np.random.randint(len(dogs))
                img = cv2.imread(dogs[i])
                if img is None
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # 固定维度，统一图像维度
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                y = 1

            else:
                i = np.random.randint(len(cats))
                img = cv2.imread(cats[i])
                if img is None
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
                y = 0

            batch_images[n] = img
            batch_label[n] = y
            n += 1
        yield batch_images,batch_label #返回批数据和标签
