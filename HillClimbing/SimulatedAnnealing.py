from random import random, randint

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def func(X, Y, x_move=1.7, y_move=1.7):
    def mul(X, Y, alis=1):
        return alis * np.exp(-(X * X + Y * Y))

    return mul(X, Y) + mul(X - x_move, Y - y_move, 2)


def show(X, Y, Z):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title("demo_hill_climbing")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow', )
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    # ax.scatter(X,Y,Z,c='r') #绘点
    plt.show()


def drawPaht(X, Y, Z, px, py, pz):
    fig = plt.figure()
    ax = Axes3D(fig)
    plt.title("demo_hill_climbing")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b' )
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')
    ax.plot(px, py, pz, 'r.')  # 绘点
    plt.show()


def hill_climb(X, Y):
    global_X = []
    global_Y = []
    # 初始温度
    temperature = 100
    # 温度下降的比率
    delta = 0.98
    # 温度精确度
    tmin = 1e-10

    len_x = len(X)
    len_y = len(Y)

    # 随机登山点
    st_x = X[0][randint(0, len_x - 1)]
    st_y = Y[randint(0, len_y - 1)][0]
    st_z = func(st_x, st_y)

    def argmax(stx, sty, alisx, alisy):
        cur = func(st_x, st_y)
        next = func(alisx, alisy)

        return cur < next and True or False

    #当温度下降到最低点之前，一直循环
    while (temperature > tmin):
        # 随机产生一个新的邻近点  与温度相关的邻近点，有可能该邻近点在定义域之外，则放弃该点
        # 说明: 温度越高幅度邻近点跳跃的幅度越大
        tmp_x = st_x + (random() * 2 - 1) * temperature
        tmp_y = st_y + + (random() * 2 - 1) * temperature
        if 4 > tmp_x >= -2 and 4 > tmp_y >= -2:
            if argmax(st_x, st_y, tmp_x, tmp_y):
                st_x = tmp_x
                st_y = tmp_y
            else:  # 有机会跳出局域最优解
                #概率变化公式
                pp = 1.0 / (1.0 + np.exp(-(func(tmp_x, tmp_y) - func(st_x, st_y)) / temperature))
                #如果产生的温度概率大于一个随机数，则接受该点
                if random() < pp:
                    st_x = tmp_x
                    st_y = tmp_y
        temperature *= delta  # 以一定的速率下降
        global_X.append(st_x)
        global_Y.append(st_y)
    print(len(global_X))
    return global_X, global_Y, func(st_x, st_y)


if __name__ == '__main__':
    X = np.arange(-2, 4, 0.1)
    Y = np.arange(-2, 4, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y, 1.7, 1.7)
    px, py, maxhill = hill_climb(X, Y)
    print(px, py, maxhill)
    drawPaht(X, Y, Z, px, py, func(np.array(px), np.array(py), 1.7, 1.7))