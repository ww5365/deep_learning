# -*- encoding: utf-8 -*-

import os
import re
'''
re： 正则表达式模块的使用

重要参考：
https://www.runoob.com/python3/python3-reg-expressions.html

'''

if __name__ == '__main__':
    '''
    re.match(pattern, string, flags=0): 
    
    功能：从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回None
    参数说明：
    flags: re.I 大小写不敏感  re.M 多行匹配   多个标识使用： |  来生效
    输出：
    匹配对象或None
    group(num = 0):
      匹配的整个表达式的字符串，group() 可以输入多个组号，它返回包含那些组所对应值的元组
      
    groups():  
       	返回一个包含所有小组字符串的元组，从 1 到 所含的小组号
    
    另外：span函数用来返回匹配对象起始位置和结束位置 
    
    '''

    print(re.match('wwW', 'www.baidu.com www.huawei.com',
                   re.I).span())  #返回[begin, end) 结束位置 (0, 3)
    print(re.match('bai', 'www.baidu.com', re.I))  # 不在起始位置匹配，返回None

    line = 'Cats are smarter than dogs'

    #  .* 匹配任意字符，至少0个字符 () 是1个匹配组  .+ 匹配任意字符，至少1个字符
    # (.*?) 是非贪婪匹配,加上? 以最少的可能性匹配 默认是贪婪方式，尽可能多的可能来匹配
    match_obj = re.match(r'(.*) are (.+?)(.+)', line, re.I)

    if match_obj:
        print(match_obj.groups())  # 返回一个包含所有小组字符串的元组
        print(match_obj.group())  # 匹配的整个表达式的字符串结果
        print(match_obj.group(1))  # 从1开始，匹配对象中的索引
        print(match_obj.group(2))
        #print(match_obj.group(3))
    else:
        print("match result is None!")
    '''
    re.research(pattern, string, flags = 0)
    
    输入和输出：类似re.match
    
    区别： 可以在任意位置开始匹配，匹配到就返回匹配对象，否则None
    
    '''
    print(re.search('BAI', 'www.baidu.com', re.I).span())  # 从bai开始匹配成功
    '''
    re.sub(pattern, repl, string, count=0, flags=0):
    功能：匹配到特定模式的字符串，并将匹配部分替换为指定的repl
    输出：处理处理后的字符串
    
    输入：
    pattern : 正则中的模式字符串。
    repl : 替换的字符串，也可为一个函数。
    string : 要被查找替换的原始字符串。
    count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。
    flags : 编译时用的匹配模式，数字形式。
    前三个为必选参数，后两个为可选参数。
    
    '''
    phone = "010-1891234567 #电话号码"

    #删除注释
    phone = re.sub(r'#.*$', '', phone)
    print(phone)

    #去掉非数字的字符
    phone = re.sub(r'\D', '', phone)
    print(phone)

    str1 = 'A23G4HFD567'
    # (?P<name> Expression) 命名捕获组 可以通过name来使用匹配组的值
    str_match = re.search(r'(?P<value>\d+)', str1)
    print(str_match.group('value'))
