# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import random

if __name__ == '__main__':

    # numpy 版本
    print(np.__version__)

    li = [[1.00, 2, 3, 4], [5, 6, 7, 8]]
    arr1 = np.array(li)
    print(li)
    print(arr1)

    print(arr1.shape)
    print(arr1.dtype)
    print(arr1.size)

    # reshape 改变数组的形状，按照列优先的方式，本质要进行深拷贝
    print("before arr1 shape:\n", arr1)
    arr2 = arr1.reshape((4, 2), order='F')
    print("after reshape arr2:\n", arr2)
    arr1[0][0] = 8.0
    print("arr1 update:\n", arr1)
    print("arr2 update or not:\n", arr2)

    # range 和 arrange 区别? arrange步长可以为小数, 且是numpy库来提供
    li2 = list(range(10))
    print(li2)
    arr3 = np.arange(0, 4, 0.5)
    print(arr3)

    # 等差，等比数列
    arr4 = np.linspace(0, 1, 11)
    print("linesapce arr4:\n", arr4)
    arr6 = np.logspace(0, 1, 3)
    print("logspace arr6:\n", arr6)

    # 特殊数组的生成接口
    arr7 = np.zeros((2, 3))
    print("zeros:\n", arr7)
    arr8 = np.eye(4)  # 对角线上全部为1
    print("eye: \n", arr8)
    arr9 = np.diag([2, 3, 4, 6])
    print("diag:\n", arr9)

    # 随机数生成

    # 使用random，只能生成1个随机数
    arr10 = random.random()  # 生成随机浮点数[0,1]
    arr11 = random.randint(1, 2)  # 生成[1,2]之间的1个数
    print(arr10, arr11)

    # 生成数组随机数？ numpy 提供接口
    arr12 = np.random.random(4)  # 生成一维，4个0-1之间的小数
    arr13 = np.random.rand(2, 3)  # 生成2*3维数据,float
    arr14 = np.random.randn(3, 2)  # 生成3*2维标准正态分布的数据
    print(arr12, arr13, arr14)

    # 切片

    li3 = [1, 2, 3, 4, 5, 6, 7]

    # list倒置
    print(li3[::-1])  # 从最后一个开始，按照步长1来返回数据
    li3.reverse()  # 直接将list中数逆置
    print(li3)

    # 二维索引切片
    arr15 = np.array([[1, 2], [3, 4], [5, 6]])
    print("arr15:", arr15[1:3, 1:2])  # 行：[1,3)  列：[1,2)

    # 改变数组的形状
    li4 = [i for i in range(10)]  # list
    print(li4)
    li5 = np.arange(12)  # array
    print(type(li5), li5)
    arr16 = li5.reshape(3, 4, order='C')
    print(arr16, arr16.ndim)

    # li6 = arr16.revel()
    li7 = arr16.flatten('C')  # 默认就是c，行优先
    li8 = arr16.flatten('F')  # 列优先
    print(li7, li8)

    # 数组拼接
    li9 = np.arange(20)
    li10 = np.arange(21, 41)
    arr17 = li9.reshape(4, 5)
    arr18 = li10.reshape(4, 5)
    print(arr17)
    print(arr18)

    arr19 = np.concatenate((arr17, arr18), axis=0)
    arr20 = np.concatenate((arr17, arr18), axis=1)

    print(arr19)
    print(arr20)

    # 排序
    li11 = np.arange(12)
    print(sorted(li11, reverse=True))

    # 数组相乘 multiply, *, dot
    arr21 = np.arange(4).reshape(2, 2)
    arr22 = np.arange(4, 8).reshape(2, 2)

    print(arr21)
    print(arr22)

    print(arr21*arr22)
    print(np.multiply(arr21, arr22))  # 对应位置相乘
    print(np.dot(arr21, arr22))  # 矩阵乘法，1维数组就是点击

    li12 = np.array([1, 2, 3])
    print(np.dot(li12, li12))  # 点积

    # 几个重要的函数
    arr24 = np.array([1, 2, 3, 8, 3, 2])
    print(np.unique(arr24))  # 去重后排序

    arr23 = np.arange(6).reshape(2, 3)
    print(arr23)

    print(np.repeat(arr23, 3, axis=0))  # 以行为维度，每行复制3次
    print(np.repeat(arr23, 3, axis=1))  # 以列为维度，每列复制3次
    print(np.tile(arr23, 3))  # 以列为维度，所有列整体复制3次

    print(np.sum(arr23, axis=0))  # 以行为维度，求和
    print(np.sum(arr23, axis=1))  # 以列为维度，求和

    print(np.mean(arr23))  # 求所有元素的均值
    print(np.std(arr23))  # 标准差 var 的平方根
    print(np.var(arr23))  # 方差
    print(np.var(arr23, axis=0))  # 方差 按照行理解，均值是每个列求均值后计算方差
    print(np.var(arr23, axis=1))  # 方差 按照列计算
    print(np.argmax(arr23))

    print(np.argmin(arr23))  # 全数组中的最小值，索引值
    # axis = 0 表示列维度，结果行结果  返回对应列中，哪行的数据最大, 索引值;每列都是从0开始
    print(np.argmax(arr23, axis=0))
    print(np.argmin(arr23, axis=1))

    print(np.cumsum(arr23)) # 累积和，斐波那契数列


