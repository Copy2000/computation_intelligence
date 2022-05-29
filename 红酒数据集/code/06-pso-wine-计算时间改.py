from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
import time
from pandas import read_csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
from matplotlib.animation import FuncAnimation

mpl.rcParams['font.sans-serif'] = ['SimHei']


class HyperparameterTuningGenetic:
    NUM_FOLDS = 5

    def __init__(self, randomSeed):
        self.randomSeed = randomSeed
        self.initWineDataset()
        self.kfold = model_selection.KFold(
            n_splits=self.NUM_FOLDS, shuffle=True, random_state=self.randomSeed)

    def initWineDataset(self):
        url = '..\data\wine.csv'

        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

    # ADABoost [n_estimators, learning_rate, algorithm]:
    # "n_estimators": integer
    # "learning_rate": float
    # "algorithm": {'SAMME', 'SAMME.R'}
    def convertParams(self, params):
        n_estimators = round(params[0])  # round to nearest integer
        learning_rate = params[1]  # no conversion needed
        # round to 0 or 1, then use as index
        algorithm = ['SAMME', 'SAMME.R'][round(params[2])]
        # print("param[2]:\t",params[2],"\talgorithm:\t",algorithm)
        return n_estimators, learning_rate, algorithm

    def getAccuracy(self, params):
        n_estimators, learning_rate, algorithm = self.convertParams(params)
        self.classifier = AdaBoostClassifier(random_state=self.randomSeed,
                                             n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             algorithm=algorithm
                                             )

        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()

    def formatParams(self, params):
        return "'n_estimators'=%3d, 'learning_rate'=%1.3f, 'algorithm'=%s" % (self.convertParams(params))


class PSO:
    def __init__(self, dimension, time, size, low, up, v_low, v_high):
        # 初始化
        self.xall = np.zeros((time + 1, size, dimension))
        self.dimension = dimension  # 变量个数
        self.time = time  # 迭代的代数
        self.size = size  # 种群大小
        self.bound = []  # 变量的约束范围
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.random = 42
        # 只有size*5的矩阵，相当于一列为一个参数的粒子
        self.x = np.zeros((self.size, self.dimension))  # 所有粒子的位置
        self.v = np.zeros((self.size, self.dimension))  # 所有粒子的速度
        self.p_best = np.zeros((self.size, self.dimension))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.dimension))[0]  # 全局最优的位置
        # 记录fitness
        self.allfitness = -1  # 记录最好个体的fitness
        self.personalfitness = []  # 记录每个个体最好的fitness 1*population
        # 当前迭代次数
        self.nowgeneration = 1

        # 初始化第0代初始全局最优解
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(
                    self.bound[0][j], self.bound[1][j])
                self.xall[0][i][j] = self.x[i][j]
                self.v[i][j] = random.uniform(self.v_low[j], self.v_high[j])
            self.p_best[i] = self.x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            self.personalfitness.append(fit)
            # 做出修改
            if fit > temp:
                self.g_best = self.p_best[i]
                self.allfitness = fit
                temp = fit

    def fitness(self, x):
        # 创建分类器
        """
        个体适应值计算
        """
        test = HyperparameterTuningGenetic(self.random)
        print("params:\t", x)
        accuracy = test.getAccuracy(x)
        print(accuracy)
        return accuracy

    def update(self, size):
        c1 = 2.0  # 学习因子
        c2 = 2.0
        w = 0.8  # 自身权重因子
        for i in range(size):
            # 更新速度(核心公式)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low[j]:
                    self.v[i][j] = self.v_low[j]
                if self.v[i][j] > self.v_high[j]:
                    self.v[i][j] = self.v_high[j]

            # 更新位置
            self.x[i] = self.x[i] + self.v[i]

            # 位置限制
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
                self.xall[self.nowgeneration][i][j] = self.x[i][j]
            # 更新p_best和g_best
            fit = self.fitness(self.x[i])
            if fit > self.personalfitness[i]:
                self.p_best[i] = self.x[i]
                self.personalfitness[i] = fit
            if fit > self.allfitness:
                self.g_best = self.x[i]
                self.allfitness = fit
            # if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
            #     self.p_best[i] = self.x[i]
            # if self.fitness(self.x[i]) > self.fitness(self.g_best):
            #     self.g_best = self.x[i]

    def pso(self):

        best = []
        self.final_best = np.array([50, 0.5, 0])
        final_best_fitness = self.fitness(self.final_best)
        start_time = time.time()
        for gen in range(self.time):
            self.update(self.size)
            if self.allfitness > final_best_fitness:
                self.final_best = self.g_best.copy()
                final_best_fitness = self.allfitness
            print("第", self.nowgeneration, "次迭代")
            self.nowgeneration += 1
            print('当前最佳位置：{}'.format(self.final_best))
            temp = final_best_fitness
            print('当前的最佳适应度：{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        end_time = time.time()
        print("time cost:\t", end_time - start_time, '\ts')

        plt.figure()
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"迭代次数")  # X轴标签
        plt.ylabel(u"适应度")  # Y轴标签
        plt.title(u"迭代过程")  # 标题
        plt.show()

    def return_result(self):
        return self.xall


if __name__ == '__main__':
    MAX_Generation = 3
    Population = 5
    dimension = 3
    v_low = [-5, -0.1, -0.5]
    v_high = [5, 0.1, 0.5]
    # [n_estimators, learning_rate, algorithm]:
    BOUNDS_LOW = [1, 0.01, 0]
    BOUNDS_HIGH = [100, 1.00, 1]
    pso = PSO(dimension, MAX_Generation, Population,
              BOUNDS_LOW, BOUNDS_HIGH, v_low, v_high)
    pso.pso()
    X_list = pso.return_result()
    # 画图==============================================
    print("begin draw pso")
    print(np.array(X_list).shape)
    save_name = str(MAX_Generation) + '_' + str(Population)
    np.save(save_name, arr=np.array(X_list))
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title('title', loc='center')
    # line = ax.plot([], [], 'b.')
    #
    # ax.set_xlim(-10, 110)
    # ax.set_ylim(-0.1, 1.1)
    # plt.ion()
    # p = plt.show()
    #
    #
    # def update_scatter(frame):
    #     i, j = frame // 10, frame % 10
    #     ax.set_title('iter = ' + str(i))
    #     X_tmp = X_list[i] + V_list[i] * j / 10.0
    #     plt.setp(line, 'xdata', X_tmp[:, 0], 'ydata', X_tmp[:, 1])
    #     return line

    # ani = FuncAnimation(fig, update_scatter, blit=True,interval=50, frames=100)
    # ani.save('5_5.gif', writer='pillow')
    # n_estimators=self.xall[]
    # learning_rate
    # algorithm
