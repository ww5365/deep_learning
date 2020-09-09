# -*- coding: utf8 -*-

import os
import sys
import random
import pathlib
'''
dir():  dir([object])  --object: 对象、变量、类型  返回：属性列表list


dir() 函数不带参数时，返回当前范围内的变量、方法和定义的类型列表；
      带参数时，返回参数的属性、方法列表。
      如果参数包含方法__dir__()，该方法将被调用。
      如果参数不包含__dir__()，该方法将最大限度地收集参数信息。

'''


class Foo(object):
    def __init__(self):
        __num = ""

    def public(self):
        print("public funciton")

    def __private(self):
        print("private function")


def get_file_path():
    '''
      os.path.dirname: 返回当前文件所在路径
      os.path.abspath: 返回当前路径的绝对路径
      os.path.split: 按照"\"来切割最后一层目录和之前的路径,tuple：比如： /tmp/test/test1  ->  (/tmp/test, test1)
      os.path.join: 将两个目录字符串拼接起来，用/进行连接
    '''
    print(os.path.dirname(__file__))
    cur_path = os.path.abspath(os.path.dirname(__file__))
    print("cur_path:", cur_path)
    print("split path:", os.path.split(cur_path))

    root_path = os.path.split(cur_path)[0]
    what_path = os.path.split(root_path)[0]  # 获取文件目录，上两层目录路径
    print("root_path:", root_path, what_path)

    work_path = os.path.join(root_path, 'Assignment3-1')
    print("work_path: ", work_path)

    # 使用pathlib直接获取文件的root路径,并加入到系统path中

    root_path = pathlib.Path(__file__).parent.parent.absolute()
    sys.path.append(sys.path.append(root_path))
    print("pathlib: syspath:", root_path, sys.path)


'''
1、几种不同的文件读取的方式
'''


def read_file():

    # 1 正常思路: 无异常处理
    root_path = pathlib.Path(__file__).parent
    file_path = os.path.join(root_path, 'test.txt')
    print("file_path:", file_path)
    f = open(file_path)
    lines = f.readlines()  # 全部读入，放入list 包含换行符
    print("file content:", lines)

    f.seek(0)  # 返回文件头
    line = f.readline()  # 读入文件的1行
    line_no = 0
    while line:
        print("line:", line)
        line = f.readline()

    f.close()  # 出异常，不会释放fp

    # 2 比较好的的思路： 有异常处理
    f2 = open(file_path)
    try:
        for line in f2.readlines():
            print("line:", line)
    except:
        print("open error")  # IOError
    finally:
        f2.close()  # 保证file会释放

    # 3 最优方案： 自带异常处理，不用close；相当于方案2，但更加精炼
    with open(file_path, 'r', encoding='utf-8') as f3:

        # readlines
        for line in f3.readlines():
            print("line3:", line)

        # readline
        f3.seek(0)
        line = f3.readline()
        while line:
            line_no = line_no + 1
            print("line33:%d -> %s" % (line_no, line))
            line = f3.readline()

        #直接读取
        f3.seek(0)
        line_no = 0
        for line in f3:
            line = line.split()
            print("line333: %d -> %s" % (line_no, line))


'''
1、创建目录：
目录文件不存在的情况下，提前创建目录，再进行目录下文件的创建，读写操作
'''


def save_file():

    # 获取当前文件目录
    cur_dir = os.path.dirname(__file__)
    print(cur_dir)

    # 创建tmp目录
    full_dir = cur_dir + "/tmp/"
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        print("full path: ", full_dir)

    file_path = full_dir + "random.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        for i in range(10):
            f.write("\t".join([str(i), random.choice(['a', 'b', 'c'])]))
            f.write('\n')


if __name__ == '__main__':

    # dir function use case
    print("\n".join(dir(Foo)))  # 4个成员函数

    get_file_path()
    read_file()
    save_file()

    # print
    str1 = "wang"
    print("format print: %s ... %s" % (str1, str1), end='*')
