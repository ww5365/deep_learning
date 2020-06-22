import numpy as np
import random

if __name__ == '__main__':
    print (np.__version__)

    li = [[2.00,1,3,4],[3,1,2,3]]
    arr1 = np.array(li)    
    print (li)
    print (arr1)

    print (arr1.shape)
    print (arr1.dtype)
    print (arr1.size)

    arr2 = arr1.reshape((4, 2), order='F')
    print (arr2)
    arr1[0][0] = 8.0
    print (arr1)
    print (arr2)

    li2 = list(range(10))
    print (li2)

    arr3 = np.arange(0,4, 0.5)
    
    print (arr3)

    arr4 = np.linspace(0,1, 11)

    print (arr4)

    arr6 = np.logspace(0,1,3)

    print("arr6:", arr6)

    arr7 = np.zeros((2,3))

    print (arr7)

    arr8 = np.eye(4)

    print (arr8)

    arr9 = np.diag([2,3,4,6])

    print (arr9)




