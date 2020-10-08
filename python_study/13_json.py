#-*- coding: utf-8 -*-

import json

'''
'''

if __name__ == '__main__':

    '''
    json文件保存本机的用户名，文件存在：欢迎该用户；文件不存在，需要接收用户输入，并保存名称。
    下次再次进入后，不需要进行输入。
    '''

    file_name = 'username.json'

    try:
        with open(file_name, 'r') as f:
            user_name = json.load(f)  ##读入json文件
    except FileNotFoundError:
        user_name = input('what is your username: ')
        with open(file_name, 'w') as f:
            json.dump(user_name, f) ##向json文件写入数据
            print("remember new user " + user_name + "!")
    else:
        print("welcome again " + user_name + '!') ## try部分代码执行成功，才会执行else，抛异常时，不执行；

    finally:
        print('this is finally!') ## 何种情况都要执行finally

        

