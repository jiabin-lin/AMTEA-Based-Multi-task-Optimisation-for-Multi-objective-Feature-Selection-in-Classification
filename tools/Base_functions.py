#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:06:30 2021

@author: linjiabin
"""

from deap import creator, tools, base, algorithms

# from deap.tools.emo import sortNondominated

import numpy as np
from tools.ProbabilisticModel import ProbabilisticModel, MixtureModel
# from deap.benchmarks.tools import hypervolume



def optimalfeatures(first_front,features):

    feat = np.array(first_front[0])
    for i in range(1,len(first_front)):
        feat += np.array(first_front[i])

    optimal_features = features[feat > 0]
    return optimal_features
# Build probabilistics model
def build_model(first_front):
    M =  np.zeros((len(first_front),len(first_front[0])))
    for i in range(len(first_front)):
        M[i,:] = first_front[i]
    model = ProbabilisticModel('umd')
    model.buildModel(M)
    return model

def build_model_m(first_front, n_c_features):
    M =  np.zeros((len(first_front), n_c_features))
    for i in range(len(first_front)):
        M[i,:] = first_front[i][:n_c_features]
    model = ProbabilisticModel('umd')
    model.buildModel(M)
    return model