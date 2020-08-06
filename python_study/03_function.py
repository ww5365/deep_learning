#-*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd

'''
yield how to work?
类似return 但又有不同; 有yield的函数，是生成器了；使用next()/send()调用执行

ref:
https://blog.csdn.net/mieleizhi0522/article/details/82142856

'''
def foo():
    print("start fun ----")
    while True:
        res = yield 4
        print("res:%d" % (res))




'''
1、lambda 表达式：
lambda 参数 ： 表达式

2、函数：
def fun():
    statement
lambda和函数区别：
1、简洁，可读性，效率？为什么效率高？
2、lamda是一次性的？在特定作用域内，类似局部变量，会被释放；
'''

def lambda_use():

    # lambda + map : 
    # 将给定的列表的值依次在所定义的函数关系中迭代并返回一个新列表
    
    li = list(map(lambda x : x * x, range(1,10)))
    print("lamda_use:", li)

    # lambda + filter :
    # filter (function, sequence)
    # 对 sequence 中的item依次执行function(item)，
    # 将结果为 True 的 item 组成一个 List/String/Tuple（取决于 sequence 的类型）并返回;

    li2 = list(filter(lambda x : x % 3 == 0, range(1,10)))
    print(li2)

    ##lambda + dataFrame + apply
    ## df.apply(function)
    ## DataFrame.apply() 函数则会遍历每一个元素，对元素运行指定的 function
    ## 参考：https://www.jianshu.com/p/4fdd6eee1b06

    matrix = [[1,2,3],[4,5,6],[7,8,9]]
    df = pd.DataFrame(matrix, columns = list('xyz'), index=['a','b','c'])
    print(df)

    ##对某列求平方
    df1 = df.apply(lambda x : (x * x ) if x.name in ['x', 'y'] else x)
    print(df1)
    
    ##对某行求平方
    df2 = df.apply(lambda x : (x * x ) if x.name== 'a' else x, axis = 1)
    print(df2)





if __name__ == '__main__':

    ##test yield
    g = foo()
    #print(g)
    print(next(g))
    print('*'*20)
    print(next(g))

    ##test lambda 
    lambda_use()

