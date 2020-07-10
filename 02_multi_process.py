# -*- coding: utf-8 -*-

import sys
from multiprocessing import Pool, Array, Value, Process


def fun(n, a):

    n.value = 2.7182
    for i in range(len(a)):
        a[i] = -a[i]
    return


if __name__ == '__main__':


    ##进程间怎么共享数据？通过 Value, Array



    '''
    共享数据类型：
    创建 num 和 arr 时使用的 'd' 和 'i' 
    参数是 array 模块使用的类型的 typecode 'd' 表示双精度浮点数， 'i' 表示有符号整数。
    这些共享对象将是进程和线程安全的。

    为了更灵活地使用共享内存，可以使用 multiprocessing.sharedctypes 模块，
    该模块支持创建从共享内存分配的任意ctypes对象。
    '''
    e = Value('d', 2.7182183)
    arr = Array('i', range(10))

    print(e.value)
    print(arr[:])

    p = Process(target=fun, args=(e, arr))

    p.start()
    p.join()

    print(e.value)
    print(arr[:])








    
