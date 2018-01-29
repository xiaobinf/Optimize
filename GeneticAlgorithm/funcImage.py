import numpy as np
import matplotlib.pyplot as plt

def func(X):
    # 函数方程
    return X+10*np.sin(5*X)+7*np.cos(4*X)

x=np.arange(0,9,0.01)
y=func(x)

def binToDec(chromosome):
    '''
    binary to decimal
    对于任何一条17位的二进制chromosome，将它复原(解码)到[0,9]这个区间中的数值
    int('',base=2) str to dec
    通用解码公式
    x = lower_bound + decimal(chromosome)×(upper_bound-lower_bound)/(2^chromosome_size-1)
    :param x:
    :return:
    '''
    x = 0 + int(str(chromosome),base=2)*(9 - 0) / (pow(2,10) - 1)
    return x

xx=[1,1,0,1,0,0,0,1,0,1]
def f(c):
    s=0
    for i in range(0,c):
        if xx[i]==1:
            s=s+pow(2,i)
    print('binToDec:', s)
    s = 0 + s * (9 - 0) / (pow(2, 10) - 1)
    print('decTo0~9:',s)
    s1 = s + 10 * np.sin(5 * s) + 7 * np.cos(4 * s);  # 计算自变量xi的适应度函数值
    # print(o,'函数值：',fitness_value[i])
    print('s1:',s1)
    return s

print(f(10))

# print(binToDec('1000000000'))
print(func(f(10)))
plt.plot(x,y)
# plt.show()