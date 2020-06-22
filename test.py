import sys
import os
import collections



if __name__ == '__main__':

    c1 = collections.defaultdict(dict)

    c1["sec1"] = {'key1', 'val1'}
    c1["sec2"] = {'key2', 'val2'}

    print (c1)

    print (c1['sec3'])


