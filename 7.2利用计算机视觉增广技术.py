#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
DATA_DIR = 'data/augmentation'
images = glob.glob(DATA_DIR+'*')

plt.figure(figsize=(10,10))
i=1
for image in images:
    img = cv2.imread(image)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(3,3,i)
    plt.imshow(img)
    i+=1
plt.show()


#定义任意变化函数，传入变换function
def plot_images(image,function,*args):
    plt.figure(figsize=(10,10))
    n_examples =3
    for i in range(n_examples):
        img = cv2.imread(image)
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=function(img,*args)
        plt.subplot(3,3,i+1)
        i+=1
    plt.show()

#变换函数，随机旋转函数操作
def rotate_image(image,rotate=20):
    width,height,_=image.shape
    random_rotation =np.random.uniform(low=-rotate,high=rotate)
    M = cv2.getRotationMatrix2D((width/2,height/2),random_rotation,1)
    return (cv2.warpAffine(image,M,(width,height)))


if __name__ == '__main__':
    plot_images(images[2],rotate_image,40)