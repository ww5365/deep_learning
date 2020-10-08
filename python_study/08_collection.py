#-*- coding: utf-8 -*-
from collections import Counter
from collections import OrderedDict
import pathlib

if __name__ == '__main__':

    print("begin to test collections") 

    ##ref: https://docs.python.org/3.7/library/collections.html

    ##collections Counter 初始化
    li = ["wang", "wei", "wang", "da", "da"]
    cnt = Counter()
    for word in li:
        cnt[word] += 1
    print(cnt) # Counter({'wang': 2, 'da': 2, 'wei': 1})

    cnt3 = Counter(li)
    print(cnt3) # 效果同上
    print(cnt3.most_common(2)) # return list: [('wang', 2), ('da', 2)]

    cnt2 = Counter(cats = 3, dogs = 2, birds = -1) 
    print(cnt2) # Counter({'cats': 3, 'dogs': 2, 'birds': -1})
    cnt2 = sorted(cnt2.elements())  #elements()会把<0的元素去掉
    print(cnt2) # ['cats', 'cats', 'cats', 'dogs', 'dogs']

    # 有序字典: 按照添加的顺序存储
    o_dict = OrderedDict()
    o_dict['1'] = 10
    o_dict['3'] = 11
    o_dict['2'] = 13

    print(o_dict['2'])



