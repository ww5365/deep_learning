#-*- coding: utf-8 -*-

import os

if __name__ == '__main__':

    # 使用rindex进行字符串的截取
    str1 = 'sz-t-xt'
    pos = str1.rindex('-')
    print('rindex pos', pos)
    print("befor pos:", str1[:pos], str1[pos + 1:].upper())

    # 内置函数的使用:
    # type(变量): 返回某个变量的类型；只认继承类自己；
    # isinstance(变量，类型): 判断某个变量类型,  返回True/False；继承类变量属于某个父类

    li1 = []
    li1.append(1)
    li1.append(2)
    print("type:isinstance", type(li1) == list, isinstance(li1, list))

    # format

    
