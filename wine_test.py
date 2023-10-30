#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this code, the classification performance on the test datasets will be calculated.
"""

import pickle

import os 
import pandas as pd

import matplotlib.pyplot as plt
# import copy
from copy import deepcopy
import numpy as np
import pickle


from sklearn.model_selection import train_test_split
from sklearn import preprocessing


from stl4 import STL # A class for evaluating the classification performance on the test datasets.

import matplotlib.pyplot as plt

import numpy as np

from deap.tools.emo import sortNondominated
from deap.benchmarks.tools import hypervolume


current_path = os.getcwd()

# Load data
wine_path = current_path + '/wine'


redwine = pd.read_csv(wine_path + '/winequality-red.csv',sep = ';')

whitewine = pd.read_csv(wine_path + '/winequality-white.csv',sep = ';')


redwine.quality.loc[redwine.quality < 7] = 0
redwine.quality.loc[redwine.quality >= 7] = 1

whitewine.quality.loc[whitewine.quality < 7] = 0
whitewine.quality.loc[whitewine.quality >= 7] = 1

dataset_1 = redwine
dataset_2 = whitewine

rand_seed = 18

X1 = dataset_1.drop(['quality'],axis=1)
X1 = pd.DataFrame(preprocessing.normalize(X1.values),columns = X1.columns)
y1 = dataset_1['quality']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=rand_seed)

X2 = dataset_2.drop(['quality'],axis=1)
X2 = pd.DataFrame(preprocessing.normalize(X2.values),columns = X2.columns)
y2 = dataset_2['quality']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=rand_seed)

p_path = 'wine'
problem = 'Wine'

X_train = [X1_train, X2_train]
y_train = [y1_train, y2_train]

X_test = [X1_test, X2_test]
y_test = [y1_test, y2_test]

stl = STL(X_train,y_train)

f = open(current_path + '/seeds-30.txt', 'r')
lines = f.readlines()
random_seeds = [int(line) for line in lines]

# HV on test set

HVT_spea2 = np.zeros((30,len(X_train)))
HVT_spea2_t = np.zeros((30,len(X_train)))

k = 0
PT_spea2 = [[] for ind in range(len(X_train))]
PT_spea2_t = [[] for ind in range(len(X_train))]

for r in random_seeds:
    
    # Load the populations obtained by the multi-task feature selection algorithm
    with open(current_path + '/' + p_path + '/source/P' + str(r) + '.pkl', 'rb') as f:
        p = pickle.load(f)
    P_spea2 = deepcopy(p)    

    with open(current_path + '/' + p_path + '/multi/P' + str(r) + '.pkl', 'rb') as f:
        p = pickle.load(f)
    P_spea2_t = deepcopy(p)
    
    prob = 1
    for p_spea2, p_spea2_t in zip(P_spea2, P_spea2_t):
        
        # Calculate classification performance on the test datasets
        p_spea2 = stl.perf_test(p_spea2, X_train[prob-1], X_test[prob-1], y_train[prob-1], y_test[prob-1])
        PT_spea2[prob-1] += deepcopy([p_spea2])
        p_spea2 = deepcopy(sortNondominated(p_spea2,k=len(p_spea2))[0])
        HVT_spea2[k,prob - 1] = hypervolume(p_spea2,[1.1,1.1])
        
        p_spea2_t = stl.perf_test(p_spea2_t, X_train[prob-1], X_test[prob-1], y_train[prob-1], y_test[prob-1])
        PT_spea2_t[prob-1] += deepcopy([p_spea2_t])
        p_spea2_t = deepcopy(sortNondominated(p_spea2_t,k=len(p_spea2))[0])
        HVT_spea2_t[k,prob - 1] = hypervolume(p_spea2_t,[1.1,1.1])
    
        
        prob += 1
    
    k += 1

# Save results
with open(current_path + '/hv_test/' + p_path + '/HVT_spea2.pkl', 'wb') as f:
    pickle.dump(HVT_spea2, f)

with open(current_path + '/hv_test/' + p_path + '/HVT_spea2_t.pkl', 'wb') as f:
    pickle.dump(HVT_spea2_t, f)













