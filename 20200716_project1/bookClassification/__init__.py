'''
@Author: your name
@Date: 2020-04-08 16:16:24
@LastEditTime: 2020-04-08 17:05:33
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /textClassification/__init__.py

__init__.py 作用：
1):将文件夹变为Python模块,Python中的每个模块的包中，都有__init__.py 文件

2):通常__init__.py 文件为空，但是我们还可以为它增加其他的功能。
   在导入一个包时，实际上是导入了它的__init__.py文件,
   这样我们可以在__init__.py文件中批量导入我们所需要的模块，而不再需要一个一个的导入。

参考：https://www.cnblogs.com/lands-ljk/p/5880483.html

主要是别人导入我的模块时，会调用,所以这里这样写，是否有问题？

eg:
__init__.py
form bookClassification import src.word2vec

别人使用时,会执行__init__.py
import bookClassifcation 

'''
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(curPath)
