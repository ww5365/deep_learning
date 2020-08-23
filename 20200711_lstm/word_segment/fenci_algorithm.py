#-*- coding: utf-8 -*-

import sys
dic_words = ['我们','学习', '人工', '智能', '未来', '是']

def word_segment_naive(input_str):
    # 第一步： 计算所有可能的分词结果，要保证每个分完的词存在于词典里，这个结果有可能会非常多。 
    len_input_str = len(input_str)
    if len_input_str == 0:  # 空字符串
        return [[]]
    else:
        result = []
        for i in range(1, len(input_str) + 1):
            if input_str[:i] in dic_words:
                for remain_segment in word_segment_naive(input_str[i:]):
                    result.append([input_str[:i]] + remain_segment)
        return result

if __name__ == '__main__':

    input_str = "我们学习人工智能人工智能是未来"

    result = word_segment_naive(input_str)

    print(result)





