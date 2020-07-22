#-*- coding: utf8 -*-

import sys



'''
1、python中_,__,__varname__,varname 下划线的前后缀，修饰变量和方法,作用？

object     # public
__object__ # special, python system use, user should not define like it
__object   # private (name mangling during runtime),类似c++种private
 _object   # obey python coding convention, consider it as private， 类似c++种protected

另注：python 没有严格的private，通过：_ClassName__object 也可访问私有成员 
根据python的约定，应该将其视作private，而不要在外部使用它们，（如果你非要使用也没辙），良好的编程习惯是不要在外部使用它。

也叫：私有变量的mangling   相当于 c++ 中宏定义： python实际讲私有变量，重替换为：_ClassName__object

'''

class Foo():
    
    def public(self):
        print("public fun")
    
    def _half_private(self):
        print("_half_private")

    def __full_private(self):
        print("__full_private")


class A(object):

    def __init__(self):
        print ("A__init__")

        self.__private()  ##name mangling后，_A__private()
        self.public()
    
    def __private(self):
        print("A.__private")

    def public(self):
        print("A.public")

class B(A):

    def __private(self):
        print("B.__private")
    
    def public(self):
        print("B.public")


if __name__ == '__main__':

    ##类成员，访问权限

    f = Foo()
    
    f.public()
    f._half_private()
    #f.__full_private() ##error 无权限
    f._Foo__full_private() ##不建议的访问方式


    ##私有变量的mangling
    print("---------------")
    b = B() ##输出什么？

    print("---------------")
    print("\n".join(dir(B)))







