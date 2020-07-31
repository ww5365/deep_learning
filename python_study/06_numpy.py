#-*-coding:utf8 -*-
import sys
import numpy as np

from multiprocessing import Array, Value

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


    '''
    3、numpy axis 的使用

    Axis就是数组层级
    设axis=i，则Numpy沿着第i个下标变化的方向进行操作 相当于sql中group by

    '''

    num1 = np.array([[1,2,3,4],[2,3,4,5]])

    print("numpy array: ", num1)

    print("numpy sum axis=1：", np.sum(num1, axis=1)) ##按照(x_i,j)j,行统计 
    print("numpy sum axis=0：", np.sum(num1, axis=0)) ##按照(x_i,j)i,列统计 

    '''
    4、numpy 计算内积 和矩阵乘法 dot
    '''

    num2 = np.array([1,2,3,1])
    num3 = np.array([[0,1,2,3],[0,1,2,0]])
    print("numpy array num2:", num2)
    
    dot_res = np.dot(num1[0], num2)
    print("numpy dot res:", dot_res)

    num3 = num3.transpose()    
    dot_res2 = np.dot(num2, num3)


    print("numpy dot res2:", dot_res2)
    
