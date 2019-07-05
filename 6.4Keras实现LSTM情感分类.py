#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

#
import os, nltk, numpy as np
import matplotlib.pyplot as plt
import collections

from keras.layers.core import Activation, Dense, Dropout, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split

# 先对文本进行分析，查看语料中有多少个独立的词，和每个句子中有多少个词
maxlen = 0
word_freqs = collections.Counter()
num_recs = 0

DATA_DIR = 'data'

ftrain = open(os.path.join(DATA_DIR, 'umich-sentiment-train.txt'), 'rb')
for line in ftrain:
    label, sentence = line.strip().split('t')
    words = nltk.word_tokenize(sentence.decode('ascii', 'ignore').lower())
    if len(words) > maxlen:
        maxlen = len(words)
    for word in words:
        word_freqs[word] += 1
    num_recs += 1

ftrain.close()
print(maxlen)  #:42
print(word_freqs)  #:2313

# 1,词典大小固定：len(word_freqs)，其他的词用伪词UNK（Unknown）替换,预测时，允许我们处理从未见过的词，把它们作为OOV(Out of Vocabulary)
# 2.固定句子长度：maxlen，不够时候全部用PAD，也就是0补齐，另外把较长句子截断，
# 3.或者根据序列把输入分成不同的批次组，
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

#创建两个词典查找表，包含两个伪词，PAD,UNK,索引按词频高到低排序，
vocab_size =min(MAX_FEATURES,len(word_freqs))+2
word2index ={x[0]:i+2 for i ,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index['PAD'] =0
word2index['UNK'] =1
index2word ={v:k for k,v in word2index.items()}

#将输入序列转换成索引序列，补足MAX_SENTENCE_LENGTH所定义的词的长度，
X =np.empty((num_recs,),dtype=list)#先生成固定长度的列表，值为空，后续赋值
y = np.zeros((num_recs,))
i =0

ftrain = open(os.path.join(DATA_DIR, 'umich-sentiment-train.txt'), 'rb')
for line in ftrain:
    label, sentence = line.strip().split('t')
    words = nltk.word_tokenize(sentence.decode('ascii', 'ignore').lower())

    seqs =[]
    for word in words:
        if word2index.has_key(word)



