#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin
from nltk import word_tokenize


def load_dataset(filename):
    '''
    下载后的文件需要预处理，先导入python，然后按照标点符号分割成不同的句子，
    并将其中无意义的字符和每个章节的标题删除，
    :return:document
    '''
    with open(file=filename,mode='r') as file:
        document =[]
        lines =file.readlines()
        for line in lines:
            #删除非内容字符
            value =clear_data(line)
            if value!='':
                #对一行文本进行分词
                for str in word_tokenize(value):
                    #跳过章节标题
                    if str =='CHAPTER':
                        break
                    else:
                        document.append(str.lower())
        return document

def clear_data(str):
    value =str.replace('\ufeff', '').replace('\n', '')
    return value

