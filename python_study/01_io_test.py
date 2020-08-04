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
      what_path = os.path.split(root_path)[0] #获取文件目录，上两层目录路径
      print("root_path:", root_path, what_path)


'''
1、几种不同的文件读取的方式
'''

def read_file():

      ##1 正常思路: 无异常处理
      f = open("/Users/wangwei69/workspace/github/tanxin/python_study/test.txt")
      lines = f.readlines() ##全部读入，放入list 包含换行符 
      print("file content:", lines)  
      
      f.seek(0) ##返回文件头
      line =f.readline() ##读入文件的1行
      line_no = 0
      while line:
            print("line:", line)
            line = f.readline() 

      f.close() ##出异常，不会释放fp

      ##2 比较好的的思路： 有异常处理
      f2 = open("/Users/wangwei69/workspace/github/tanxin/python_study/test.txt")
      try:
            for line in f2.readlines():
                  print("line:", line)
      except:
            print ("open error") ##IOError
      finally:
            f2.close() ##保证file会释放

      ##3 最优方案： 自带异常处理，不用close；相当于方案2，但更加精炼
      with open('/Users/wangwei69/workspace/github/tanxin/python_study/test.txt', 'r', encoding= 'utf-8') as f3:
            for line in f3.readlines():
                  print("line3:", line)
            f3.seek(0)
            line = f3.readline()
            while line:
                  print("line33:", line)
                  line = f3.readline()

if __name__ == '__main__':
      
      ##dir function use case
      print("\n".join(dir(Foo))) ## 3个成员函数
      get_file_path()
      read_file()