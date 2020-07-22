#-*-coding:utf8 -*-
import sys
import numpy as np

if __name__=='__main__':

    '''
    1、 使用numpy产出出：随机数
    '''
    
    # randint(low=x, high=y, size=z or (m,n), d) 产出[low, high)之间的随机值，产出个数z或（m,n）

    indcies1 = np.random.randint(low = 0, high = 4, size = 10)
    indcies2 = np.random.randint(low = 0, high = 4, size = (2,3))
    
    print(indcies1)
    print(indcies2)
    
    tmp1 = np.random.random(3) ##float类型随机数
    tmp2 = np.random.uniform(low = 0/10, high = 1/10, size = 5) ##[0, 0.1)之间的均匀分布
    
    print(tmp1)
    print(tmp2)

    '''
    2、numpy类型数组，转为ctypes类型

    '''
    ##ctypes 类型使用
    tmp = np.ctypeslib.as_ctypes(tmp2)
    print("ctypes tmp:",tmp)   ##<c_double_Array_5 object at 0x119c8cb00>
    print("dir tmp:", dir(tmp), tmp._type_)
    tmp = Array(tmp._type_, tmp, lock=False)
    print("Array tmp:", tmp)  ##<c_double_Array_5 object at 0x11a4ce050> 实际都是c double array类型，不过Array后，可以控制进线程的安全性；
    print(tmp[0])