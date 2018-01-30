import numpy as np
import random
import matplotlib.pyplot as plt

# f(x) = x+10*sin(5*x)+7*cos(4*x), x∈[0,9]

# population是population_size*chromosome_size的二维数组
# 迭代次数generation_size
generation_size=100
gn=0
population=np.zeros((900,10),int)
population_new=np.zeros((900,10),int)
upper_bound=9
lower_bound=0
fitness_value = [0] * np.shape(population)[0]
fitness_sum=[0]*900
fitness_average=[0]*generation_size
best_fitness=0
best_generation=0
best_individual=list()

# 该函数调试时使用
# def fiwr():
#     global population
#     global fitness_value
#     with open('a.txt','a+') as f:
#         for line in population:
#             f.write(str(line)+'\n')
#         f.write(str(fitness_value))

def init(population_size, chromosome_size):
    '''
    随机初始化种群
    :param population_size:种群个体个数
    :param chromosome_size: 种群个体的长度
    :return: 返回种群二维数组
    '''
    global population
    for i in range(0,population_size):
        for j in range(0,chromosome_size):
            population[i][j]=random.choice([0,1])

def fitness(population_size, chromosome_size):
    '''
    计算适应度
    :param population_size:
    :param chromosome_size:
    :return:
    '''
    global population
    global fitness_value
    global lower_bound
    global upper_bound
    for i in range(0,population_size):
        fitness_value[i]=0
    # o=1


    for i in range(0,population_size):
        for j in range(0,chromosome_size):
            if population[i][j]==1:
                # 个体的染色体序列排序，小端排序
                fitness_value[i]=fitness_value[i]+pow(2,j) #将二进制转为十进制
        fitness_value[i] = lower_bound + fitness_value[i] * (upper_bound-lower_bound)/(pow(2,chromosome_size) - 1)  #将十进制投影到0-9区间
        # fitness_value[i]=0 + fitness_value[i]*(9 - 0) / (pow(2,10) - 1)
        fitness_value[i] = fitness_value[i] + 10 * np.sin(5 * fitness_value[i]) + 7 * np.cos(4 * fitness_value[i]); #计算自变量xi的适应度函数值


    # # 打印population
    # for i in range(0,population_size):
    #     print(o,'population[{}]'.format(i),str(population[i]))
    #     print(o,'fitness_value[{}]'.format(i),fitness_value[i])
    #     o+=1

# def binListToDec(binList):
#     '''
#     将一个二进制列表转为十进制整数
#     :param binList:
#     :return:
#     '''
#     dec=0
#     for i in range(0,len(binList)):
#         if binList[i]==1:
#             dec=dec+pow(2,i)
#     return dec

def rank(population_size, chromosome_size):
    '''
    对个体按适应度大小进行排序，并且保存最佳个体
    :param population_size:
    :param chromosome_size:
    :return:
    '''
    global fitness_sum
    global population
    global best_fitness
    global best_generation
    global best_individual
    global gn
    # 初始化fitness_sum的值
    for i in range(0,population_size):
        fitness_sum[i]=0

    for i in range(0,population_size):
        # 冒泡排序
        min_index=i
        for j in range(i+1,population_size):
            if fitness_value[j]<fitness_value[i]:
                # 交换
                fitness_value[i], fitness_value[j] = fitness_value[j], fitness_value[i]
                for k in range(0,chromosome_size):
                    population[i][k],population[j][k]=population[j][k],population[i][k]

    # fitness_sum(i) = 前i个个体的适应度之和
    for i in range(0,population_size):
        if i==1:
            fitness_sum[i]=fitness_sum[i]+fitness_value[i]
        else:
            fitness_sum[i]=fitness_sum[i-1]+fitness_value[i]
     # 第gn次迭代个体的平均适应度
    fitness_average[gn]=fitness_sum[population_size-1]/population_size

    if fitness_value[population_size-1]>best_fitness:
        # 当数据比较小时，很多时候前几次迭代就已经找到了个体最优解，不断的迭代只是将种群整体的平均解向最优解靠近
        best_fitness=fitness_value[population_size-1]
        best_generation = gn
        print(best_generation,gn)
        best_individual=population[population_size-1]
    gn+=1

def selection(population_size, chromosome_size, elitism):
    '''
    遗传算法的选择操作  选一些适应度为正数的个体出来
    :param population_size:种群大小
    :param chromosome_size:个体染色体长度
    :param elitism:是否精英选择
    :return:
    '''
    global population
    global population_new
    global fitness_sum   #种群累积适应度
    for i in range(0,population_size):
        r=random.random()*fitness_sum[population_size-1]
        first=0
        last=population_size-1
        mid=round((last+first)/2)
        idx=-1
        # 排中法选择个体
        while first<last and idx==-1:
            if r > fitness_sum[mid]:
                first = mid
            elif r < fitness_sum[mid]:
                last = mid
            else:
                idx = mid
                break

            mid = round((last + first) / 2)
            if (last - first) == 1:
                idx = last
                break
        # 产生新个体
        for j in range(0,chromosome_size):
            population_new[i][j]=population[idx][j]

    # 是否是精英选择
    if elitism:
        p=population_size-1
    else:
        p=population_size
    #  如果精英选择，将population中前population_size-1个个体更新，最后一个最优个体保留
    for i in range(0,p):
        for j in range(0,chromosome_size):
            population[i][j]=population_new[i][j]

def crossover(population_size, chromosome_size, cross_rate):
    '''
    相邻个体的交叉操作
    :param population_size:
    :param chromosome_size:
    :param cross_rate:
    :return:
    '''
    global population
    for i in range(0,population_size,2):
        if random.random()<cross_rate:
            cross_position=round(random.random()*chromosome_size)
            if cross_position==chromosome_size or cross_position==0:
                # 数组越界 完全交换没有意义
                continue
            for j in range(cross_position,chromosome_size):
                population[i][j],population[i+1][j]=population[i+1][j],population[i][j]

def mutation(population_size, chromosome_size, mutate_rate):
    '''
    变异操作  单点变异
    :param population_size:
    :param chromosome_size:
    :param mutate_rate:
    :return:
    '''
    global population
    for i in range(0,population_size):
        if random.random()<mutate_rate:
            mutate_position=round(random.random()*chromosome_size)
            if mutate_position==chromosome_size:
                # 越界 不变异
                continue
            population[i][mutate_position]=1-population[i][mutate_position]

def plotGA(generation_size):
    '''
    绘制平均适应度迭代过程
    :param generation_size:
    :return:
    '''
    X=list(range(generation_size))
    Y=fitness_average
    plt.plot(X,Y)
    plt.show()


def genetic_algorithm(population_size, chromosome_size, generation_size, cross_rate, mutate_rate, elitism):
    '''
    # Genetic Algorithm for Functional Maximum Problem
    :param population_size:输入种群大小
    :param chromosome_size: 输入染色体长度
    :param generation_size: 输入迭代次数
    :param cross_rate: 输入交叉概率
    :param mutate_rate: 输入变异概率
    :param elitism: 是否是精英选择
    :return:(m,n,p,q)
    m:输出最佳个体
    n:输出最佳适应度
    p:输出最佳个体出现的迭代次数
    '''
    global gn               # 当前迭代次数
    global fitness_value    # 当前代适应度矩阵
    global best_fitness     # 历代最佳适应值
    global best_individual  # 历代最佳个体
    global best_generation  # 最佳个体出现代
    global upper_bound      # 自变量的区间上限
    global lower_bound       # 自变量的区间下限
    global fitness_average  # 构建迭代次数×1的零矩阵

    # 初始化种群
    init(population_size,chromosome_size)

    for i in range(0,generation_size):
        # 迭代gn次
        fitness(population_size, chromosome_size)                   # 计算适应度
        rank(population_size, chromosome_size)                      # 对个体按适应度大小进行排序
        selection(population_size, chromosome_size, elitism)        # 选择操作
        crossover(population_size, chromosome_size, cross_rate)     # 交叉操作
        mutation(population_size, chromosome_size, mutate_rate)     # 变异操作

    plotGA(generation_size)             # 打印最优适应度迭代过程

    m = best_individual                 # 获得最佳个体
    n = best_fitness                    # 获得最佳适应度
    p = best_generation                 # 获得最佳个体出现时的迭代次数

    print('fitness_sum:',str(fitness_sum))
    print('fitness_average:', str(fitness_average))
    return m,n,p


print(genetic_algorithm(900,10,100,0.6,0.01,True))