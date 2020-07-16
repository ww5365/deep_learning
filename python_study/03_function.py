#-*- coding: utf-8 -*-
import sys

'''
yield how to work?
类似return 但又有不同

'''
def foo():
    print("start fun ----")
    while True:
        res = yield 4
        print("res:%d" % (res))




if __name__ == '__main__':

    g = foo()
    print(g)
    print(next(g))

    ##test yield

