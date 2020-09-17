# -*- coding: utf-8 -*-

import sys
from multiprocessing import Pool, Array, Value, Process

from multiprocessing import Queue

import time


def fun(n, a):

    n.value = 2.7182
    for i in range(len(a)):
        a[i] = -a[i]
    return


def producer(q):
    try:
        q.put(1, block=False)  #block
        print('producer put num 1')
        time.sleep(5)
        print('producer finished!')
    except:
        print("producer except")
        pass


def consumer(q, idx):
    try:
        res = q.get(block=True, timeout=1)  #等待1s后，没有数据来，抛出异常Queue.empty
        print('consumer[%d] res:%d' % (idx, res))
    except:
        print("consumer except")
        pass


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

    ##多线程的调用
    p = Process(target=fun, args=(e, arr))
    p.start()
    p.join()
    print(e.value)
    print(arr[:])
    '''
    使用Queue 实现生产者和消费者
    
    Queue是多进程安全的队列，可以使用Queue实现多进程之间的数据传递。
    put方法用以插入数据到队列中，put方法还有两个可选参数：blocked和timeout。
    如果blocked为True（默认值），并且timeout为正值，该方法会阻塞timeout指定的时间，直到该队列有剩余的空间。
    如果超时，会抛出Queue.Full异常。如果blocked为False，但该Queue已满，会立即抛出Queue.Full异常。
 
    get方法可以从队列读取并且删除一个元素。同样，get方法有两个可选参数：blocked和timeout。
    如果blocked为True（默认值），并且timeout为正值，那么在等待时间内没有取到任何元素，会抛出Queue.Empty异常。
    如果blocked为False，有两种情况存在，如果Queue有一个值可用，则立即返回该值，否则，如果队列为空，则立即抛出Queue.Empty异常。
    
    '''
    max_size = 100
    q = Queue(max_size)
    p1 = Process(target=producer, args=(q, ))  # args 只有1个参数时候，逗号不要漏
    p1.start()

    thread = 3
    for idx in range(thread):
        p2 = Process(target=consumer, args=(q, idx))
        p2.start()
        p2.join()

    p1.join()  #阻塞时间较长，等待p1结束后，才结束主进程

    print("main thread finished!"
