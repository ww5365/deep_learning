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

    # list 使用 []定义
    li = [1,2,3,4]
    
    # 负数索引, -1 表示获取最后一个元素
    print(li[-1], li[-3])

    # 增，删，修改

    li.append(8)
    li.append(9)  # 尾部插入1个元素
    li.insert(0, 19) # 下标0处，插入元素19
    print(li)

    del li[1] # 删除下标索引1的值
    x1 = li.pop() # 尾部元素弹出，删除
    x2 = li.pop(0) # 头部元素，即下标0处，弹出删除
    li.remove(3)  #删除元素值为3的元素

    print(li, x1, x2)  ## 体会一下三种删除方式的异同？ del pop remove

    #排序

    li2 = sorted(li, reverse = True) ## li 中元素位置不变
    print("sorted:", li2, li)
    li.sort(reverse = True)  ## li中元素也永久发生变化

    # 翻转

    print("before reverse: ", li)
    li.reverse()
    print("after reverse: ", li)

    # 数组越界
    try:
        print(li[-1])  ##较好的获取最后一个元素的方式，只有为空时才抛出异常
        print(li[4])
    except :
        print("error")
        pass


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
        print("key:value=[%d:%s]"%(key,value)) ##key:value=[1:test1]

    for idx, value in enumerate(zip(li1,li2)):
        print("key:value=[%d:%s]"%(idx,value)) ## key:value=[0:(1, 'test1')]

    '''
    3. set
    集合（set）是一个无序的不重复元素序列。
    可以使用大括号 { } 或者 set() 函数创建集合
    注意：创建一个空集合必须用 set() 而不是 { }，因为 { } 是用来创建一个空字典

    重复的key，只保留一个

    '''  
    set1 = set()
    set2 = {'wang', 'wei'}  #集合初始化，和定义一个空dict不同 
    set1.add('test') # 只能添加1个元素
    
    #增加
    set1.update(set2) #也可以添加多个元素，且参数可以是列表，元组，字典等
    set1.update({'wang':5, 'num2':6}) #仅添加key到set中，update和之前重复的,仅保留1个，比如wang
    print(set1)
    #删除
    set1.remove("wei") #key不存在的话，会抛出异常
    set1.discard("we") #key不存在的话，不会抛出异常
    print(set1)

    #查询
    elem = "wang"
    if elem in set1:
      print(elem)



    