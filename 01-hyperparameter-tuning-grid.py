import numpy as np
import time
import random

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from pandas import read_csv
from evolutionary_search import EvolutionaryAlgorithmSearchCV
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


class HyperparameterTuningGrid:
    NUM_FOLDS = 5

    def __init__(self, randomSeed):

        self.randomSeed = randomSeed
        self.initWineDataset()
        self.initClassifier()
        self.initKfold()
        self.initGridParams()

    def initWineDataset(self):
        #url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        url = 'D:\大三下\计算智能\第二次\wine.csv'
        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

    def initClassifier(self):
        self.classifier = AdaBoostClassifier(random_state=self.randomSeed)

    def initKfold(self):
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, shuffle=True,
                                           random_state=self.randomSeed)

    def initGridParams(self):
        '''
        n_estimators 在10到100之间线性间隔10个值进行测试
        learning_rate 在对数0.1和10之间对数间隔10个值进行测试
        测试algorithm的两个可能值
        涵盖200个（10*10*2）不同的网格组合
        '''
        self.gridParams = {
            'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'learning_rate': np.logspace(-2, 0, num=10, base=10),
            'algorithm': ['SAMME', 'SAMME.R'],
        }

    # 使用accuracy的平均值指标来评估分类器的默认超参数值的准确性
    def getDefaultAccuracy(self):
        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')
        return cv_results.mean()

    # 对之前定义的一组测试过的超参数值进行常规的网格搜索，根据参数的k-fold交叉验证均值准确性指标就可以确定最佳参数组合
    def gridTest(self):
        global score1, learning_rate1, n_estimators1, score2, learning_rate2, n_estimators2
        print("performing grid search...")
        start = time.time()
        gridSearch = GridSearchCV(estimator=self.classifier,
                                  param_grid=self.gridParams,
                                  cv=self.kfold,
                                  # n_jobs=4,
                                  scoring='accuracy')

        gridSearch.fit(self.X, self.y)
        print("best parameters: ", gridSearch.best_params_)
        print("best score: ", gridSearch.best_score_)
        end = time.time()
        print("Time Elapse:", end - start)
        score1 = []
        learning_rate1 = []
        n_estimators1 = []
        score2 = []
        learning_rate2 = []
        n_estimators2 = []

        for mean_, params_ in zip(gridSearch.cv_results_['mean_test_score'], gridSearch.cv_results_['params']):
            print("%0.3f for %r" % (mean_, params_))
            if (params_['algorithm'] == 'SAMME'):
                score1.append(mean_)
                learning_rate1.append(params_['learning_rate'])
                n_estimators1.append(params_['n_estimators'])

            if (params_['algorithm'] == 'SAMME.R'):
                score2.append(mean_)
                learning_rate2.append(params_['learning_rate'])
                n_estimators2.append(params_['n_estimators'])
        mpl.rcParams['legend.fontsize'] = 10
        fig1 = plt.figure()
        ax1 = fig1.gca(projection='3d')
        ax1.plot(score1, learning_rate1, n_estimators1, label='SAMME')
        ax1.set_xlabel("score")
        ax1.set_ylabel("learning rate")
        ax1.set_zlabel("n_estimators")
        ax1.legend()
        plt.show()
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.plot(score2, learning_rate2, n_estimators2, label='SAMME.R')
        ax2.set_xlabel("score")
        ax2.set_ylabel("learning rate")
        ax2.set_zlabel("n_estimators")
        ax2.legend()
        plt.pause(0.001)
        plt.ion()
        # plt.show()

    def geneticGridTest(self):
        avg = []
        min = []
        print("performing Genetic grid search...")
        start = time.time()
        gridSearch = EvolutionaryAlgorithmSearchCV(estimator=self.classifier,
                                                   params=self.gridParams,
                                                   cv=self.kfold,
                                                   scoring='accuracy',
                                                   verbose=True,
                                                   # n_jobs=4,
                                                   population_size=20,
                                                   gene_mutation_prob=0.5,
                                                   tournament_size=2,
                                                   generations_number=10)
        gridSearch.fit(self.X, self.y)
        end = time.time()
        print("Time Elapsed = ", end - start)
        # print(gridSearch.best_score_)
        # print(gridSearch.all_logbooks_)
        # print(gridSearch.all_logbooks_[0][0])
        for i in range(len(gridSearch.all_logbooks_[0])):
            avg.append(gridSearch.all_logbooks_[0][i]['avg'])
            min.append(gridSearch.all_logbooks_[0][i]['min'])
        fig,axes = plt.subplots(figsize=(12, 6))
        axes.plot(avg)
        axes.plot(min)
        axes.legend(["avg","min"],loc=0)
        axes.set_title("genetic search")
        plt.show()

def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create a problem instance:
    test = HyperparameterTuningGrid(RANDOM_SEED)

    # ===================default params=================
    # print("Default Classifier Hyperparameter values:")
    # print(test.classifier.get_params())
    # print("score with default values = ", test.getDefaultAccuracy())

    # ==============网格搜索==================
    print()
    test.gridTest()
    print()

    # ===========遗传算法==================
    test.geneticGridTest()


if __name__ == "__main__":
    main()
    # print("===============score==============")
    # print(score1)
    # print("============learning rate=============")
    # print(learning_rate1)
    # print("=============n_estimators=============")
    # print(n_estimators1)


'''
网格搜索的最优策略
best parameters:  {'algorithm': 'SAMME', 'learning_rate': 0.3593813663804626, 'n_estimators': 60}
best score:  0.9720634920634922
Time Elapse: 92.37686347961426


遗传算法网格搜索：只有10次迭代，每次20个个体
Best individual is: {'n_estimators': 100, 'learning_rate': 0.3593813663804626, 'algorithm': 'SAMME'}
with fitness: 0.9719101123595506
Time Elapsed =  22.87787914276123
'''

