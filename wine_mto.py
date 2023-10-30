#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 10:45:27 2021

@author: linjiabin

This code is about running the multi-task SPEA2 feature selection algorithm on the Wine datasets.
"""


import os 
import pandas as pd
# import random

# import time

# import matplotlib.pyplot as plt

# from copy import deepcopy
# import numpy as np
# import pickle


from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# from tools.Base_functions import build_model

# from tools.mto3 import MTO

# import sys

from tools.Go_MTO3 import go_mto # Run the multi-task optimisation (mto) framework




def main():


    current_path = os.getcwd()
    wine_path = current_path + '/wine'
    
    # Load data
    redwine = pd.read_csv(wine_path + '/winequality-red.csv',sep = ';')
    redwine.columns
    redwine.head(5)
    print(redwine.describe())
    
    
    whitewine = pd.read_csv(wine_path + '/winequality-white.csv',sep = ';')
    whitewine.columns
    whitewine.head(5)
    print(whitewine.describe())
    
    
    redwine.quality.loc[redwine.quality < 7] = 0
    redwine.quality.loc[redwine.quality >= 7] = 1
    
    whitewine.quality.loc[whitewine.quality < 7] = 0
    whitewine.quality.loc[whitewine.quality >= 7] = 1
    
    # Two related datasets corresponding to two related feature selection tasks
    dataset_1 = redwine
    dataset_2 = whitewine
    
    rand_seed = 18
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    X1 = dataset_1.drop(['quality'],axis=1)
    X1 = pd.DataFrame(preprocessing.normalize(X1.values),columns = X1.columns)
    y1 = dataset_1['quality']
    
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3, random_state=rand_seed)
    
    X2 = dataset_2.drop(['quality'],axis=1)
    X2 = pd.DataFrame(preprocessing.normalize(X2.values),columns = X2.columns)
    y2 = dataset_2['quality']
    
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=rand_seed)
    
    
    X = [X1_train,X2_train]
    Y = [y1_train,y2_train]
    
    # Run the multi-task SPEA2 feature selection algorithm on two related datasets
    go_mto(X,Y,'wine')

if __name__ == "__main__":
    main()  

    
  
    
    
    
    
    








