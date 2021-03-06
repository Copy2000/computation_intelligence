from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
import time
from pandas import read_csv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']

class HyperparameterTuningGenetic:
    
    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initAdultDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True,random_state=self.randomSeed)

    def initAdultDataset(self):
        url = 'processed_adult.csv'

        self.data = read_csv(url, header=None, usecols=range(0, 15))
        self.X = self.data.iloc[:, 0:13]
        self.y = self.data.iloc[:, 14]
        # self.data = read_csv(url, header=None, usecols=range(0, 12))
        # self.X = self.data.iloc[:, 0:10]
        # self.y = self.data.iloc[:, 11]

    # ADABoost [n_estimators, learning_rate, algorithm]:
    # "n_estimators": integer
    # "learning_rate": float
    # "algorithm": {'SAMME', 'SAMME.R'}
    def convertParams(self, params):
        #print("\n_estimators    learning_rate   algorithm\n",params)
        n_estimators = round(params[0])  # round to nearest integer
        learning_rate = params[1]        # no conversion needed
        algorithm = ['SAMME', 'SAMME.R'][round(params[2])]  # round to 0 or 1, then use as index
        return n_estimators, learning_rate, algorithm

    def getAccuracy(self, params):
        n_estimators, learning_rate, algorithm = self.convertParams(params)
        n_estimators=int(n_estimators)
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
        # ?????????
        self.dimension = dimension  # ????????????
        self.time = time  # ???????????????
        self.size = size  # ????????????
        self.bound = []  # ?????????????????????
        self.bound.append(low)
        self.bound.append(up)
        self.v_low = v_low
        self.v_high = v_high
        self.random = 42
        #??????size*5???????????????????????????????????????????????????
        self.x = np.zeros((self.size, self.dimension))  # ?????????????????????
        self.v = np.zeros((self.size, self.dimension))  # ?????????????????????
        self.p_best = np.zeros((self.size, self.dimension))  # ???????????????????????????
        self.g_best = np.zeros((1, self.dimension))[0]  # ?????????????????????

        # ????????????0????????????????????????
        temp = -1000000
        for i in range(self.size):
            for j in range(self.dimension):
                self.x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.v[i][j] = random.uniform(self.v_low[j], self.v_high[j])
            self.p_best[i] = self.x[i]  # ?????????????????????
            fit = self.fitness(self.p_best[i])
            # ????????????
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, x):
        # ???????????????
        """
        ?????????????????????
        """
        test = HyperparameterTuningGenetic(self.random)
        accuracy = test.getAccuracy(x)
        print(accuracy)
        return accuracy



    def update(self, size):
        c1 = 1  # ????????????
        c2 = 1
        w = 0.8  # ??????????????????
        for i in range(size):
            # ????????????(????????????)
            self.v[i] = w * self.v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # ????????????
            for j in range(self.dimension):
                if self.v[i][j] < self.v_low[j]:
                    self.v[i][j] = self.v_low[j]
                if self.v[i][j] > self.v_high[j]:
                    self.v[i][j] = self.v_high[j]

            # ????????????
            self.x[i] = self.x[i] + self.v[i]
            # ????????????
            for j in range(self.dimension):
                if self.x[i][j] < self.bound[0][j]:
                    self.x[i][j] = self.bound[0][j]
                if self.x[i][j] > self.bound[1][j]:
                    self.x[i][j] = self.bound[1][j]
            # ??????p_best???g_best
            if self.fitness(self.x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.x[i]
            if self.fitness(self.x[i]) > self.fitness(self.g_best):
                self.g_best = self.x[i]

    def pso(self):

        best = []
        i=1
        self.final_best = np.array([1, 0.5, 0])
        start_time=time.time()
        for gen in range(self.time):
            self.update(self.size)
            if self.fitness(self.g_best) > self.fitness(self.final_best):
                self.final_best = self.g_best.copy()
            print("the",i,"generation")
            i+=1
            print('?????????????????????{}'.format(self.final_best))
            temp = self.fitness(self.final_best)
            print('???????????????????????????{}'.format(temp))
            best.append(temp)
        t = [i for i in range(self.time)]
        end_time=time.time()
        print("time cost:\t",end_time-start_time)
        plt.figure()
        plt.plot(t, best, color='red', marker='.', ms=15)
        plt.rcParams['axes.unicode_minus'] = False
        plt.margins(0)
        plt.xlabel(u"????????????")  # X?????????
        plt.ylabel(u"?????????")  # Y?????????
        plt.title(u"????????????")  # ??????
        plt.show()


if __name__ == '__main__':
    MAX_Generation = 10
    Population = 10
    dimension = 3
    # v_low = -1
    # v_high = 1
    v_low = [-1,-0.01,-0.1]
    v_high = [1,0.01,0.1]
    # [n_estimators, learning_rate, algorithm]:
    BOUNDS_LOW =  [  1, 0.01, 0]
    BOUNDS_HIGH = [100, 1.00, 1]
    pso = PSO(dimension, MAX_Generation, Population, BOUNDS_LOW, BOUNDS_HIGH, v_low, v_high)
    pso.pso()

