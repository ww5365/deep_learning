#-*-coding:utf8 -*-
import sys



'''
常用数据结构的使用技巧和方法

'''

if __name__ == '__main__':

    ##list 取出符合条件的元素的下标indices
    li = [1,2,3,3,4,5,7,8,10]
    indices = [idx for idx in range(len(li)) if li[idx]%2 == 0]
    print("indices:", indices)