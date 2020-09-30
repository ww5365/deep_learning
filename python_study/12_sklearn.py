# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

'''
StandardScaler
作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本。
训练集可用；之后，测试集合可用同样的接口去转换，保持一致；

【注：】
并不是所有的标准化都能给estimator带来好处。
“Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).”

参考: 
https://scikit-learn.org/stable/modules/preprocessing.html
https://blog.csdn.net/u012609509/article/details/7855470

'''


def pre_data_process():

    # StandardScaler

    np.random.seed(123)
    train = np.random.randn(5, 4)
    print(train)

    # 使用numpy统计
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    var = std * std
    print("train mean:{} std:{} var:{}".format(mean, std, var))

    # 使用StandardScaler进行样本的标准化：去均值和方差归一化

    scaler = StandardScaler()
    scaler.fit(train)  # 计算
    final_train = scaler.transform(train)  # 转成标准化样本
    print("scaler info: mean:{} var:{}".format(scaler.mean_, scaler.var_))
    print("-----------scaler result------------")
    print(final_train)

    # 验证StandardScaler完成的计算

    print("-----------verify scaler------------")
    train1 = train - mean  # 广播
    print(train1)
    train1 = train1/std
    print(train1)  # 这个等于final_train结果


def tf_idf():
    doc = ['I have a pen', 'I have an big egg']
    model = TfidfVectorizer().fit(doc)  # 计算文档doc的tfidf模型

    sparse_res = model.transform(doc)  # 模型内容转成sparse方式，可视化

    print(model.vocabulary_)  # 生成的词典
    print(sparse_res)
    print(sparse_res.todense())  # 稠密表示方法,词向量的表示方法，按照词典的顺序来表示

    print(sparse_res.toarray())


if __name__ == '__main__':

    tf_idf()
    pre_data_process()
