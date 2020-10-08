#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt 

if __name__ == '__main__':

    print("this is test for {key}".format(key="matplotlib"))

    # 折线图
    squares = [x**2 for x in range(10)]
    plt.plot(squares, linewidth = 2)
    plt.title("squares function", fontsize = 24)
    plt.xlabel("x", fontsize = 14)
    plt.ylabel("y", fontsize = 14)
    #plt.show()

    # 散点图

    x_values = list(range(-100, 101))
    y_values = [x**2 for x in x_values]

    ## 参数s，表示点的大小,默认蓝点，蓝轮廓; c 点的颜色参数；edagecolor点轮廓颜色
    #plt.scatter(x_values, y_values, s=5, edgecolor='none', c='red')
    
    # 参数c可以是RGB的三元组
    #plt.scatter(x_values, y_values, s=5, edgecolor='none', c=(0,0,0.8))
    
    ## 颜色映射 c=y_values cmap参数使用蓝色
    plt.scatter(x_values, y_values, s=5, edgecolors='none', c=y_values, cmap=plt.cm.Blues)
    plt.axis([-110, 110, 0, 11000])  ##设置xy轴的最大和最小值
    #plt.show()

    plt.savefig('test_plot.png', bbox_inches='tight')




     