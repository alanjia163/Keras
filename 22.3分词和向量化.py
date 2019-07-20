#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
文本导入术后，需要将每个单词 包括标点符号转换成数字，作为输入，
分词后，使用Gensim将单词转换为整数,并生成一个简单的向量

'''

import nltk
import ssl

# 取消ssl认证
from gensim import corpora

ssl._create_default_https_context = ssl._create_unverified_context
#下载nltk数据包
nltk.download()

def word_to_integer(document):
    dic =corpora.Dictionary([document])
    #保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    #将单词转换为整数
    values =[]
    for word in document:
        values.append(dic_set[word])

    return values


