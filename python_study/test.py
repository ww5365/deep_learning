import sys
import os
import collections
import math
import numpy as np
from multiprocessing import Array




if __name__ == '__main__':

    ## 文件读写方式1

    fi = open('./README.md', 'r')
    lineNo = 0
    for line in fi:
        line = line.strip()
        lineNo += 1
        print ("lineNo:%d content:%s"%(lineNo, line))
    file_bytes = fi.tell() ##除了1行，每行的换行符计算1个字节

    print("file bytes: %d" % file_bytes)

    ##存放多个dict？使用collections
    c1 = collections.defaultdict(dict)
    c1["sec1"] = {'key1', 'val1'}
    c1["sec2"] = {'key2', 'val2'}
    print (c1)
    print (c1['sec3']) ##访问没有元素不会抛异常

    ##list 使用
    vocab_size = 10
    parent = [1] * (2 * vocab_size - 2) ##向量的填充操作？
    print("parent: ", parent)
    ###list 中冒号使用说明 [start:end:step] 取[start,end)之间的元素，同时step为步长
    li = [0,1,2,3,4]
    print(li[::2]) 


    ##定义并初始化字典
    result_dict = {1 : 4}
    for key in result_dict.keys():
        print (key)

    ##字符串分割
    line = "wang  wei nihao  a 北京"
    tokens = line.split(' ') #默认split()也是空格分割；但和split(' ')有区别，对于连续空格情况，这个只考虑1个，其它空格保留
    for token in tokens:
        print ("line token: %s" %token)


    ##enumerate 枚举迭代器； 逆序访问数组或字符串;
    for index,key in enumerate(li[::-1]):
        print("index:%d value:%d" %(index,key))


    ##直接两个list进行cat，相加
    count = [t for t in li] + [1e15] * (5 - 1)
    print(count)

    for idx in range(4):
        print(idx)

    ##math.power(x,y) x^y 
    count  = 100
    norm = math.pow(count, 0.5)
    print(norm)

    # 随机数产出
    # randint(low=x, high=y, size=z or (m,n), d) 产出[low, high)之间的随机值，产出个数z或（m,n）

    indcies1 = np.random.randint(low = 0, high = 4, size = 10)
    indcies2 = np.random.randint(low = 0, high = 4, size = (2,3))
    
    print(indcies1)
    print(indcies2)
    
    
    tmp1 = np.random.random(3)
    tmp2 = np.random.uniform(low = 0/10, high = 1/10, size = 5)
    
    print(tmp1)
    print(tmp2)

    ##ctypes 类型使用
    tmp = np.ctypeslib.as_ctypes(tmp2)
    print("ctypes tmp:",tmp)   ##<c_double_Array_5 object at 0x119c8cb00>
    print("dir tmp:", dir(tmp), tmp._type_)
    tmp = Array(tmp._type_, tmp, lock=False)
    print("Array tmp:", tmp)  ##<c_double_Array_5 object at 0x11a4ce050>
    print(tmp[0])




    

    





    
    






