#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

from keras.layers import Input
from keras.layers.core import Activation,Dense,Dropout,Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add,concatenate,dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import collections
import itertools
import nltk
import numpy as np
import matplotlib.pyplot as plt
import os

'''
给定的故事和问题，预测答案，两输入一个输出


'''
DATA_DIR = "./data"

TRAIN_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_test.txt")


def get_data(infile):
    '''
    第一个问题回答任数据包含1000个短句，一个故事包含两到三个句子，后面跟着一个问题，答案附加在故事的末尾，
    把文件解析到故事，问题和答案组成的三元组
    :param infile:
    :return: stories,questions,answers
    '''
    stories,questions,answers=[],[],[]
    story_text =[]
    fin =open(TRAIN_FILE,'rb')
    for line in fin:
        line =line.decode('utf-8').strip()
        lno,text = line.split('',1)
        if 't' in text:
            question,answer ,_ =text.split('t')
            stories.append(story_text)
            questions.append(question)
            answers.append(answer)
            story_text=[]
        else:
            story_text.append(text)

        fin.close()
        return stories,questions,answers
# get the data
data_train = get_data(TRAIN_FILE)
data_test = get_data(TEST_FILE)
print(len(data_train[0]), len(data_test[0]))


def build_vocab(train_data, test_data):
    '''
    构建字典，字典较小，只有22个独立词
    :param train_data:
    :param test_data:
    :return:
    '''
    counter = collections.Counter()
    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            for sent in story:
                for word in nltk.word_tokenize(sent):
                    counter[word.lower()] += 1
        for question in questions:
            for word in nltk.word_tokenize(question):
                counter[word.lower()] += 1
        for answer in answers:
            for word in nltk.word_tokenize(answer):
                counter[word.lower()] += 1
    # no OOV here because there are not too many words in dataset
    word2idx = {w:(i+1) for i, (w, _) in enumerate(counter.most_common())}
    word2idx["PAD"] = 0
    idx2word = {v:k for k, v in word2idx.items()}
    return word2idx, idx2word
# build vocabulary from all the data
word2idx, idx2word = build_vocab(data_train, data_test)
vocab_size = len(word2idx)
print("vocab size: {:d}".format(len(word2idx)))

def get_maxlens(train_data, test_data):
    '''
    找出故事和问题最大长度，发现故事长度最大是14个词，问题最大长度是4个词
    :param train_data:
    :param test_data:
    :return:
    '''
    story_maxlen, question_maxlen = 0, 0
    for stories, questions, _ in [train_data, test_data]:
        for story in stories:
            story_len = 0
            for sent in story:
                swords = nltk.word_tokenize(sent)
                story_len += len(swords)
            if story_len > story_maxlen:
                story_maxlen = story_len
        for question in questions:
            question_len = len(nltk.word_tokenize(question))
            if question_len > question_maxlen:
                question_maxlen = question_len
    return story_maxlen, question_maxlen
# compute max sequence length for each entity
story_maxlen, question_maxlen = get_maxlens(data_train, data_test)
print("story maxlen: {:d}, question maxlen: {:d}".format(story_maxlen, question_maxlen))

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    '''
    RNN输入是一个词ID序列，因此使用字典将三元组，故事问题答案，转换成词ID，
    不足长度的用'PAD，就是0补齐，
    :param data:
    :param word2idx:
    :param story_maxlen:
    :param question_maxlen:
    :return:
    '''
    Xs, Xq, Y = [], [], []
    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [[word2idx[w.lower()] for w in nltk.word_tokenize(s)]
                                   for s in story]
        xs = list(itertools.chain.from_iterable(xs))
        xq = [word2idx[w.lower()] for w in nltk.word_tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        Y.append(word2idx[answer.lower()])
    return pad_sequences(Xs, maxlen=story_maxlen),\
           pad_sequences(Xq, maxlen=question_maxlen),\
           np_utils.to_categorical(Y, num_classes=len(word2idx))
# vectorize the data
Xstrain, Xqtrain, Ytrain = vectorize(data_train, word2idx, story_maxlen, question_maxlen)
Xstest, Xqtest, Ytest = vectorize(data_test, word2idx, story_maxlen, question_maxlen)
print(Xstrain.shape, Xqtrain.shape, Ytrain.shape, Xstest.shape, Xqtest.shape, Ytest.shape)




'''
model有两个输入：问题词ID序列，故事词序列
嵌入层将词ID换成64维度向量，将其转换成大小为max_question_length向量，
'''

# define network
EMBEDDING_SIZE = 64
LATENT_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 10

# inputs
story_input = Input(shape=(story_maxlen,))
question_input = Input(shape=(question_maxlen,))

# story encoder memory
story_encoder = Embedding(input_dim=vocab_size,
                         output_dim=EMBEDDING_SIZE,
                         input_length=story_maxlen)(story_input)
story_encoder = Dropout(0.3)(story_encoder)

# ######################question encoder
question_encoder = Embedding(input_dim=vocab_size,
                            output_dim=EMBEDDING_SIZE,
                            input_length=question_maxlen)(question_input)
question_encoder = Dropout(0.3)(question_encoder)


# 使用dot（），函数 match between story and question
match = dot([story_encoder, question_encoder], axes=[2, 2])