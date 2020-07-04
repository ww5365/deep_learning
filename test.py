import sys
import os
import collections



if __name__ == '__main__':

    ##存放多个dict？使用collections
    c1 = collections.defaultdict(dict)
    c1["sec1"] = {'key1', 'val1'}
    c1["sec2"] = {'key2', 'val2'}
    print (c1)
    print (c1['sec3']) ##访问没有元素不会抛异常

    ##
    vocab_size = 10
    parent = [1] * (2 * vocab_size - 2) ##向量的填充操作？
    print("parent: ", parent)


    ##定义并初始化字典
    result_dict = {1 : 4}
    for key in result_dict.keys():
        print (key)

    






