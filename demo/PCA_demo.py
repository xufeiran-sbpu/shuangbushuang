#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import itertools
import bisect
import matplotlib.pyplot as plt

sys.path.append('../')
from loglizer.models import PCA
from loglizer import dataloader, preprocessing

# np.set_printoptions(threshold=np.inf)

struct_log = '../data/HDFS/HDFS_100k.log_structured.csv' # The structured log file
label_file = '../data/HDFS/anomaly_label.csv' # The anomaly label file
def calculatShapley(cFunction,coalition,nPlayer):
    coalition=list(coalition)
    for i in range(0,len(coalition)):
        coalition[i]=list(coalition[i])


    print("start calculate shapley:")
    shapley_values = []
    for i in range(len(nPlayer)):
        shapley = 0
        for j in coalition:
            if i not in j:
                j=list(j)
                cmod = len(j)
                Cui = j[:]
                bisect.insort_left(Cui,i)
                l = coalition.index(j)
                k = coalition.index(Cui)
                temp = float(float(cFunction[k]) - float(cFunction[l])) *\
                           float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
                shapley += temp
                # if i is 0:
                #     print j, Cui, cmod, n-cmod-1, characteristic_function[k], characteristic_function[l], math.factorial(cmod), math.factorial(n - cmod - 1), math.factorial(n)

        cmod = 0
        Cui = [i]
        k = coalition.index(Cui)
        temp = float(cFunction[k]) * float(math.factorial(cmod) * math.factorial(len(nPlayer) - cmod - 1)) / float(math.factorial(len(nPlayer)))
        shapley += temp

        shapley_values.append(shapley)

    return (shapley_values)

def getcoaltionlist():
    coalition=[]
    for i in range(1, 15):
        for p in itertools.combinations((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), i):
            coalition.append(p)
            # if i==2:
            #     print(p)
    return coalition

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(struct_log,
                                                                label_file=label_file,
                                                                window='session', 
                                                                train_ratio=0.5,
                                                                split_type='uniform')
    # print("load_HDFS后的x_train",x_train,type(x_train))
    # print("y_train!!!!!!!!:",y_train)
    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf', 
                                              normalization='zero-mean')
    # print("fit_transform后的x_train 尺寸是：",x_train.shape)
    x_test = feature_extractor.transform(x_test)
    # print("输入后的测试数据矩阵:",x_test)
    # print("----", x_train.shape)
    # print("****",x_test.shape)

    model = PCA()
    model.fit(x_train)
    shap_X_test = x_test
    print('Train validation:')

    precision, recall, f1 = model.evaluate(x_train, y_train)
    # print("训练出来的y是这个样子：", y_train)
    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)
    print('Test shapley validation:')
    for k in range(0,np.shape(x_test)[1]):
        shap_X_test = np.copy(x_test)
        shap_X_test[:,k] = np.zeros(np.shape(shap_X_test[:,k]))
        precision, recall, f1 = model.evaluate(shap_X_test, y_test)

    Eventname = ['E5', 'E22', 'E11', 'E9', 'E26', 'E2', 'E7', 'E3', 'E10', 'E14', 'E6', 'E16', 'E18', 'E25']
    all = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    coalition = getcoaltionlist()
    cvalue = []
    print("coalition number:", len(coalition))

    print('calculate characteristic function for coalition')
    for i in range(0, len(coalition)):
        shap_X_test = np.copy(x_test)
        shap_X_train = np.copy(x_train)
        diff = set(all) ^ set(coalition[i])
        if (len(diff) != 0):
            shap_X_test[:, tuple(diff)] = np.zeros(np.shape(shap_X_test[:, tuple(diff)]))
            shap_X_train[:, tuple(diff)] = np.zeros(np.shape(shap_X_train[:, tuple(diff)]))

        model.fit(shap_X_train)
        # model.fit(shap_X_train, y_train)
        precision, recall, f1 = model.evaluate(shap_X_test, y_test)

        cvalue.append(precision)

    print("V1-14:", cvalue[0:14])
    print("V-1:", cvalue[-1])

    print('prepared characteristic value')
    shapleys = calculatShapley(cvalue, coalition, all)
    print(shapleys)

    plt.bar(range(14), shapleys, color='lightsteelblue')
    plt.plot(range(14), shapleys, marker='o', color='coral')  # coral
    plt.xticks(range(14), Eventname)
    plt.xlabel('Event')
    plt.ylabel("Shapley addictive index")
    plt.legend()
    plt.show()