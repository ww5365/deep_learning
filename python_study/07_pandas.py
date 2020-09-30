# -*-coding: utf8-*-
import numpy as np
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

    # DataFrame 使用
    li = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    df1 = pd.DataFrame(li)  # 默认index是0,1，2...colums也是
    print(df1)

    # 默认index是0,1，2...colums设定
    df2 = pd.DataFrame(li, index=[1, 2, 3], columns=['col1', 'col2', 'col3'])
    print("----------------")
    print(df2)
    print(df2.dtypes)
    print(df2.columns)
    print(df2.index)
    print(df2.values)
    print("shape:", df2.shape)  # 维度,  返回tuple (rows, cols)
    print(df2.size)
    print(df2.ndim)

    # 列简单索引
    print("----------------")
    print(df2['col2'])
    print(df2[['col1', 'col2']])  # 多列索引

    # 行索引
    print(df2[[False, True, True]])
    print(df2[df2['col1'] > 4])

    # 单值和多值索引，使用loc,iloc, 同时涉及行列索引的，建议用loc和iloc

    print("----------------")
    print(df2)

    print(df2.loc[2, 'col2'])
    print(df2.loc[[1, 2], ['col1', 'col3']])
    print(df2.loc[[1, 3], 'col1':'col2'])  # 使用[]可以挑选某行或列
    print(df2.loc[1:3, 'col1':'col2'])  # 使用：是连续的切片

    print(df2.iloc[1:3, [0, 2]])  # 使用行列index进行切片

    # 增和更新操作: 一般先索引，再赋值
    df2.loc[1, 'col1'] = 3
    print(df2)

    # 删除: 原series不变 , 只能用名称索引删除

    print("drop----------------")
    df3 = df2.drop(2)  # 删除第2行
    print(df2)
    print(df3)

    df4 = df2.drop('col2', axis=1)
    print(df2)
    print(df4)

    df4.drop('col1', axis=1, inplace=True)  # 在原来结构中直接删除 inplace 参数
    print(df4)

    print("----------------")
    # 首尾几行
    print(df2.tail(1))

    # 整表统计
    # axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    # axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵

    print("statistic----------------")
    print(df2)
    print(df2.describe().round(2).T)  # 整体统计describe, T 转置， round(N):四舍五入，小数位数

    print(df2.loc[[2, 3], :].mean(axis=1))
    print(df2.loc[[2, 3], :].var(axis=1))

    df2['col4'] = list('王者之')
    print(df2)
    print(df2.loc[[2, 3], :].mean(axis=1))
    print(df2.loc[[2, 3], :].var(axis=1))

    # 将某列转为list
    print("-----------------")
    print(df2)
    print(df2['col4'].tolist())

    # 根据列长，生成colN 的列名
    col_name = ["col" + str(i+1) for i in range(df2.shape[1])]
    print(col_name)

    df3 = df2[col_name]
    print(df2[col_name], type(df3))

    # 读入文件：保存类型：dataframe
    # data1 = pd.read_csv(file_path1, '\t')
    # print(data1)
    # data2 = pd.read_csv(file_path2, '\t')
    # print(data2)

    li1 = ['中国你好这本书', '文学', '']
    li2 = ['math统计原理这本书', '数学', '']

    # 默认的是逐列来创建，怎么逐行创建dataframe数据? [li] 转成二维的

    data1 = pd.DataFrame([li1], columns=['title',  'desc', 'text'])
    data2 = pd.DataFrame([li2], columns=['title',  'desc', 'text'])

    # 新增加一行数据到dataframe
    print("---------------------------")
    dict1 = {'title': ['竞走'], 'desc': ['体育'], 'text': ['']}  # 注意value需要加[]
    print(pd.DataFrame(dict1))
    # 需要ignore_index, 但为什么没有生效？？
    data1.append(pd.DataFrame(dict1), ignore_index=True)
    print("append----", data1)

    # 两个dataframe合并
    print("---------------------------")
    data = pd.concat([data1, data2])  # list中的多个dataframe拼接[d1,d2],列名只保留第1个
    print(data)

    data['text'] = data['title'] + data['desc']
    # print(data['text'])

    # 使用lambda和函数达到的效果一致;每句话进行切词
    data['text'] = data['text'].apply(lambda doc: list(jieba.cut(doc)))
    # data['text'] = data['text'].apply(query_cut)

    # 将dataframe中'text'列，逐行处理，分词后的结果，使用空格分割
    data['text'] = data["text"].apply(lambda doc: " ".join(doc))
    print('cut word string: ', data['text'])

    # 提取tfidf的特征
    model = TfidfVectorizer()
    tfidf = model.fit(data['text'])
    # print(tfidf.vocabulary_)
    # print(model.transform(data['text']))

    # 每行使用空格分好词的文本，转成list; eg: A B C -> [A,B,C]
    data['text'] = data['text'].apply(lambda x: x.split(' '))
    print("cut word to list: ", data['text'])

    print("-------------------------------------------------")
    # dataframe.text ?  二维[[A,B][C,D]]矩阵

    '''
    了解下gensim： https://zhuanlan.zhihu.com/p/37175253
    Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达;

    '''

    id2word = gensim.corpora.Dictionary(data.text)  # 建立语料特征的索引字典
    # print(id2word)
    corpus = [id2word.doc2bow(text) for text in data.text]  # 词袋模型的稀疏向量
    print('corpus: ', corpus)
    print(id2word.token2id)  # token到id映射
