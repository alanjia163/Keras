#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Author: Jia ShiLin

'''
分词后将标点符号排除，生成词云，查看下哪些词最频繁，生成词云使用pyecharts
WordCloud类，传入单词列表和单词出现的频率

'''




#生词词云
from gensim import corpora
from pyecharts.charts import WordCloud


def show_word_cloud(document):
    letf_words=['.',',','?','!',';',':','\'','(',')']

    #生成字典
    dic =corpora.Dictionary([document])
    #计算每个词使用频率
    words_set = dic.doc2bow(document)

    #生成单词列表和使用频率列表
    words,frequences =[],[]
    for item in words_set:
        key = item[0]
        frequence =item[1]
        if word not in letf_words:
            words.append(word)
            frequences.append(frequences)

    #使用pyecharts生成词云
    word_cloud =WordCloud(width =1000,height =620)
    word_cloud.add(series_name='Alice\'s word cloud',attr=words,value =frequences,word_size_range=[20,100])
    word_cloud.render()

