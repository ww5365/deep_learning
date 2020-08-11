#-*- coding: utf-8 -*-
'''
@Author: xiaoyao jiang
@Date: 2020-04-08 17:22:54
@LastEditTime: 2020-07-18 09:50:43
@LastEditors: xiaoyao jiang
@Description: train embedding & tfidf & autoencoder
@FilePath: /bookClassification/src/word2vec/embedding.py
'''
import pandas as pd
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from gensim.models import LdaMulticore
from gensim.models.ldamodel import LdaModel
import gensim

from __init__ import *
from src.utils.config import root_path
from src.utils.tools import create_logger, query_cut
from src.word2vec.autoencoder import AutoEncoder
logger = create_logger(root_path + '/logs/embedding.log')


'''
1. 类定义，没用object 使用type？ 区别？

参考：
https://wiki.jikexueyuan.com/project/explore-python/Class/metaclass.html

2. 单例实现


'''
class SingletonMetaclass(type):
    '''
    @description: singleton
    '''
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass,
                                    self).__call__(*args, **kwargs)
            return self.__instance
        else:
            return self.__instance

class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        '''
        @description: This is embedding class. Maybe call so many times. we need use singleton model.
        In this class, we can use tfidf, word2vec, fasttext, autoencoder word embedding
        @param {type} None
        @return: None
        '''
        # 停止词
        self.stopWords = open(root_path + '/data/stopwords.txt', encoding='utf-8').readlines()
        # autuencoder
        self.ae = AutoEncoder()

    def load_data(self):
        '''
        @description:Load all data, then do word segment
        @param {type} None
        @return:None
        '''
        logger.info('load data')

        #https://www.cnblogs.com/qi-yuan-008/p/12410354.html concat read_csv 参考a
        #dataFrame 类型
        self.data = pd.concat([
            pd.read_csv(root_path + '/data/train.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/dev.tsv', sep='\t'),
            pd.read_csv(root_path + '/data/test.tsv', sep='\t')
        ])
        self.data["text"] = self.data['title'] + self.data['desc']
        self.data["text"] = self.data["text"].apply(query_cut)
        self.data['text'] = self.data["text"].apply(lambda x: " ".join(x))##保存了使用空格分割的切词结果

    def trainer(self):
        '''
        @description: Train tfidf,  word2vec, fasttext and autoencoder
        @param {type} None
        @return: None
        '''

        logger.info('train tfidf')
        '''
        TfidfVectorizer:使用方法

        stop_words: 单词的列表，eg： ['word1', 'word2']
        max_df : 最大文档频率，eg：总共5个文档，出现在其中4个文档，df=0.8 超过就过滤
        min_df : 同上
        ngram_range: tuple,(1,2)可以是单个词，也可是两个词 进入到统计范畴


        fit(): 
        Fit the vectorizer/model to the training data and save the vectorizer/model 
        to a variable (returns sklearn.feature_extraction.text.TfidfVectorizer)
        输入形式：
        * ['doc1', 'doc2', ..] docN都是分好词的，使用空格分割；list;
        * dataframe 某个列字段

        transform(): 
        Use the variable output from fit() to transformer validation/test data 
        (returns scipy.sparse.csr.csr_matrix)
        
        vocabulary_: 词语与列的对应关系

        ref:
        https://stackoverflow.com/questions/53027864/what-is-the-difference-between-tfidfvectorizer-fit-transfrom-and-tfidf-transform
        https://blog.csdn.net/blmoistawinde/article/details/80816179
        '''
        count_vect = TfidfVectorizer(stop_words=self.stopWords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.data["text"])  #直接支持了dataframe列索引的数据格式

        '''
        Word2Vec 参数说明：
        sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
        size是输出词向量的维数，值太小会导致词映射因为冲突而影响结果，值太大则会耗内存并使算法计算变慢，一般值取为100到200之间。
        window是句子中当前词与目标词之间的最大距离，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
        min_count是对词进行过滤，频率小于min-count的单词则会被忽视，默认值为5。(0-100)
        sample(float) - 用于配置哪些较高频率的词随机下采样的阈值，有用范围是（0,1e-5）
        hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
        workers,使用线程数。控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。

        train:

        total_examples (int) – 统计句子数量
        report_delay (float) – 进度报告等待的秒数

        ref：https://www.jianshu.com/p/b996e7e0d0b0

        '''


        logger.info('train word2vec')
        self.data['text'] = self.data["text"].apply(lambda x: x.split(' ')) ##[[word1,word2],[wordn..]]
        self.w2v = models.Word2Vec(min_count=2,
                                   window=5,
                                   size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   iter=30,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.data["text"])
        self.w2v.train(self.data["text"],
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)

        logger.info('train fast')
        # 训练fast的词向量
        self.fast = models.FastText(
            self.data["text"],
            size=300,  # 向量维度
            window=3,  # 移动窗口
            alpha=0.03,
            min_count=2,  # 对字典进行截断, 小于该数的则会被切掉,增大该值可以减少词表个数
            iter=30,  # 迭代次数
            max_n=3,
            word_ngrams=2,
            max_vocab_size=50000)


        '''
        了解下gensim： https://zhuanlan.zhihu.com/p/37175253
        Gensim是一款开源的第三方Python工具包，用于从原始的非结构化的文本中，无监督地学习到文本隐层的主题向量表达;

        1、corpora.Dictionary 对象 : word <-> id 之间的映射
        Dictionary(documents=None, prune_at=2000000)
        字典封装了在归一化词汇（word）与整型id之间的映射关系。
        主要函数有 doc2bow，它将许多词汇转换成词袋（bag-of-words）模型表示：一个2-tuples列表（word_id, word_frequency）。
        如果给定了documents，使用它们进行字典初始化（参见：add_documents()）

        输入：data.text  二维[[doc1_A,doc1_B][doc2_C,doc2_D]]矩阵 

        2、j 
        https://radimrehurek.com/gensim/models/ldamulticore.html

        '''
        logger.info('train lda')
        self.id2word = gensim.corpora.Dictionary(self.data.text)  #建立语料特征的索引字典
        corpus = [self.id2word.doc2bow(text) for text in self.data.text] #词袋模型的稀疏向量
        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha='asymmetric')
        ##lstm
        logger.info('train autoencoder')
        self.ae.train(self.data)

    def saver(self):
        '''
        @description: save all model
        @param {type} None
        @return: None
        '''
        logger.info('save autoencoder model')
        self.ae.save()

        logger.info('save tfidf model')
        joblib.dump(self.tfidf, root_path + '/model/embedding/tfidf')

        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(root_path +
                                         '/model/embedding/w2v.bin',
                                         binary=False)

        logger.info('save fast model')
        self.fast.wv.save_word2vec_format(root_path +
                                          '/model/embedding/fast.bin',
                                          binary=False)

        logger.info('save lda model')
        self.LDAmodel.save(root_path + '/model/embedding/lda')

    def load(self):
        '''
        @description: Load all embedding model
        @param {type} None
        @return: None
        '''
        logger.info('load tfidf model')
        self.tfidf = joblib.load(root_path + '/model/embedding/tfidf')

        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/w2v.bin', binary=False)

        logger.info('load fast model')
        self.fast = models.KeyedVectors.load_word2vec_format(
            root_path + '/model/embedding/fast.bin', binary=False)

        logger.info('load lda model')
        self.lda = LdaModel.load(root_path + '/model/embedding/lda')

        logger.info('load autoencoder model')
        self.ae.load()


if __name__ == "__main__":
    em = Embedding()
    em.load_data()
    em.trainer()
    em.saver()
