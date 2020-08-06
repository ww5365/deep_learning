#-*-coding: utf8-*-

import pandas as pd
import os 
import sys
import jieba
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer



file_path = '/Users/wangwei69/workspace/github/tanxin/python_study/test.txt'
file_path1 = '/Users/wangwei69/workspace/github/tanxin/20200716_project1/bookClassification/data/train.csv'
file_path2 = '/Users/wangwei69/workspace/github/tanxin/20200716_project1/bookClassification/data/test.csv'
file_path3 = '/Users/wangwei69/workspace/github/tanxin/20200716_project1/bookClassification/data/dev.csv'

def query_cut(query):
    '''
    @description: word segment 分词
    @param {type} query: input data
    @return:
    list of cut word
    '''
    return list(jieba.cut(query))

if __name__ == '__main__':

    ##读入文件：保存类型：dataframe
    data1 = pd.read_csv(file_path1, '\t')
    #print(data1)
    data2 = pd.read_csv(file_path2, '\t')
    #print(data2)

    print("---------------------------")
    data = pd.concat([data1, data2]) ##list中的多个dataframe拼接[d1,d2],列名只保留第1个
    print(data)
    
    data['text'] = data['title'] + data['desc']
    #print(data['text'])
    
    ##使用lambda和函数达到的效果一致;每句话进行切词
    data['text'] = data['text'].apply(lambda doc : list(jieba.cut(doc)))
    #data['text'] = data['text'].apply(query_cut)

    ##将dataframe中'text'列，逐行处理，分词后的结果，使用空格分割
    data['text'] = data["text"].apply(lambda doc : " ".join(doc))
    print(data['text'])

    ##提取tfidf的特征
    model = TfidfVectorizer()
    tfidf = model.fit(data['text'])
    #print(tfidf.vocabulary_)
    #print(model.transform(data['text']))

    ##每行使用空格分好词的文本，转成list; eg: A B C -> [A,B,C]
    data['text'] = data['text'].apply(lambda x : x.split(' '))
    print(data['text'])

    print("-------------------------------------------------")
    ##dataframe.text ?  二维[[A,B][C,D]]矩阵 

    for test in data.text:
        #print(test)
        print()

    '''
    了解下gensim： https://zhuanlan.zhihu.com/p/37175253
    Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达;

    '''

    id2word = gensim.corpora.Dictionary(data.text) #建立语料特征的索引字典
    corpus = [id2word.doc2bow(text) for text in data.text] #词袋模型的稀疏向量
    print(corpus)
    print(id2word.token2id)








