#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Jia ShiLin
'''
本模块用来服务于7.2视觉增广函数
'''
import cv2

import numpy as np


def adjust_brightness(image, brightness=60):
    '''
    调整亮度
    '''
    rand_brightness = np.random.uniform(low=-brightness, high=brightness)
    return (cv2.add(image, rand_brightness))


def random_shifts(image, shitf_max_x=100, shift_max_y=100):
    '''
    提供参数随机移动图片
    :param image:
    :param shitf_max_x:
    :param shift_max_y:
    :return:
    '''
    width, height, _ = image.shape
    shitf_x = np.random.randint(shift_max_x)
    shitf_y = np.random.randint(shift_max_y)
    M = np.float32([1, 0, shitf_max_x], [0, 1, shift_max_y])
    return (cv2.warpAffine(image, M,(height,width)))


def random_flip(image,p_flip=0.5):
    rand = np.random.randint()
    if rand<p_flip:
        image=cv2.flip(image,1)
    return image
