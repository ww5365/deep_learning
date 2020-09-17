#-*- encoding: utf-8 -*-
import os
import json

if __name__ == '__main__':
    print("map use example")

    # dict 初始化
    dict1 = dict()
    dict2 = {}
    dict3 = {'id': 12, 'name': 'ww', 'class': 5}

    # 更新
    dict1.update(dict3)
    dict2['id'] = 13
    print("dict1: dict3", dict1, dict3)

    # 查
    print(dict1.get('id', 0))
    print(dict1.get('ids', 0))
    print(dict1['id'])

    # 轮询： items() 返回[(key, val)..] 元祖
    dict2['id1'] = 14
    dict2['id2'] = 10

    for key, val in dict2.items():
        print("dict2 key:val", key, val)

    #按照第2列排序
    res = sorted(dict2.items(), key=lambda x: x[1])  # sorted返回list
    for key, val in dict2.items():
        print("dict2 sorted key:val", key, val)
    for key, val in res:
        print("res sorted key:val", key, val)

    # 把map转成json

    dict4 = {}
    dict4['id'] = [1, 2, 3, 4]
    dict4['name'] = 'ww'
    dict4['doc'] = {'doc1': 'test1', "doc2": 'test2'}

    print(dict4)

    print(json.dumps(dict4))
