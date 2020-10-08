#-*- coding: utf-8 -*-

import os
import string


'''
string: 字符串的使用
'''

def basic_fun():

    str1 = "love china's lanD !"
    print(str1)

    str_lower = str1.lower()
    str_upper = str1.upper()
    str_title = str1.title() # 首字母大写

    print("lower: %s upper: %s title: %s" % (str_lower, str_upper, str_title))


    # 将数字转化为字符串输出
    msg = 'your number is: '
    num = 10
    print(msg + str(num))





if __name__ == '__main__':

    basic_fun()