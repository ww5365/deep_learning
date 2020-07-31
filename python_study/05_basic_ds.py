#-*-coding:utf8 -*-
import sys



'''
常用数据结构的使用技巧和方法

'''

if __name__ == '__main__':


    '''
    1、 list 常见的使用

    可重复，类型可不同

    '''
    
    ##list 取出符合条件的元素的下标indices
    li = [1,2,3,3,4,5,7,8,10]
    indices = [idx for idx in range(len(li)) if li[idx]%2 == 0]
    print("indices:", indices)

    ##list 填充操作？
    vocab_size = 10
    parent = [1] * (2 * vocab_size - 2) 
    print("parent: ", parent)

    ###list 中冒号使用说明 [start:end:step] 取[start,end)之间的元素，同时step为步长 (注意end是闭区间，不取这个位置上数)
    li = [0,1,2,3,4]
    print(li[::2])

    '''
    2、元组 (x,y) 
      tuple与list类似，不同之处在于tuple中的元素不能进行修改。
      而且tuple使用小括号()，list使用方括号[]。
      * 比列表操作速度快
      * 对数据写保护
      * 可用于字符串格式化中
      * 可作为字典的key

    zip:用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。 
    python3中返回的是：一个对象，可以手动转成list
    '''
    li1 = [1,2,4,5,2]
    li2 = ["test1","test2","test4","test5","test2"]

    tuple_list = zip(li1, li2)
    print("tuple_list:", tuple_list) ##一个对象

    for key, value in zip(li1,li2): ##一个可迭代对象，对每个元素(是一个元祖)
        print("key:value=[%d:%s]"%(key,value))

    for idx, value in enumerate(zip(li1,li2)):
        print("key:value=[%d:%s]"%(idx,value))
    