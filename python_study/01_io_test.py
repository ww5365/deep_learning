#-*- coding: utf8 -*-

import os
import sys



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
      print(os.path.dirname(__file__))
      cur_path = os.path.abspath(os.path.dirname(__file__))
      root_path = os.path.split(cur_path)[0]
      what_path = os.path.split(root_path)[0]
      print("root_path:", root_path, what_path)


if __name__ == '__main__':
      
      ##dir function use case
      print("\n".join(dir(Foo))) ## 3个成员函数

      get_file_path()