"""
The framework of multi-task SPEA2 feature selection algorithm.
Input: related datasets, the name of datasets
"""

import os 
# import pandas as pd
import random

import time

# import matplotlib.pyplot as plt

from copy import deepcopy
import numpy as np
import pickle


# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing

from tools.Base_functions import build_model

from tools.mto3 import MTO
# from deap.tools.emo import sortNondominated
# from deap.benchmarks.tools import hypervolume
from deap import creator, tools, base, algorithms

# import sys


def go_mto(X,Y,Task_name):
    
    current_path = os.getcwd()
    Task_path = current_path + '/' + Task_name
    # Collect features of each dataset
    Features =[]
    for x in X:
        Features += deepcopy([x.columns])
    
                
    # Build mappings between the features of the related feature selection tasks
    Mappings = [[] for ind in X]
    
        
    for i in range(len(X)-1):
        for j in range(i+1,len(X)):
            Common_features_i_j = deepcopy(list(set(Features[i]).intersection(Features[j])))
            # Common_features_j_i = deepcopy(list(set(Features[j]).intersection(Features[i])))
            
            T_inds = []
            S_ins = []
            for f in Common_features_i_j:
                T_inds += [deepcopy(list(Features[i]).index(f))]
                S_ins += [deepcopy(list(Features[j]).index(f))] 
            # Mappings[i][]
            Mappings[i] += deepcopy([[T_inds,S_ins]])
            Mappings[j] += deepcopy([[S_ins,T_inds]])
            

    # load 30 random seeds    
    f = open(current_path + '/seeds-30.txt', 'r')
    lines = f.readlines()
    random_seeds = [int(line) for line in lines]
    

    run = 0

    creator.create("FitnessMulti", base.Fitness, weights = (-1.0, -1.0))
    creator.create("Individual", list, fitness = creator.FitnessMulti)

    # Run the algorithm on the benchmark datasets 30 times corresponding to 30 different random seeds
    for r in random_seeds:
        print('Run: ', run)
        global random_seed
        random_seed = r
        # random_seed = 321
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Calculate the computational time of each run
        start = time.time()
        
        
        Populations = [] # Store the population at each generation
        HVs = [[] for x in X] # Store the hypervolume obtained at each generation
        Trs = [[] for x in X] # Store the transfer coefficient obtained at each generation
        MTOs = [] # Store the multi-task optimisation classes for each feature selection tasks
        i = 0
        for x, y in zip(X, Y):
            mto = MTO(x, y, random_seed, creator)
            # Initialize the population of each feature selection task
            population = mto.toolbox.population()
            # Evaluate the individuals of each population
            fits = mto.toolbox.map(mto.toolbox.evaluate, population)
            for fit, ind in zip(fits, population):
                ind.fitness.values = fit
            Populations += deepcopy([population])
            MTOs += deepcopy([mto])                     
            i += 1

            
        # Run the algorithm for generations    
        Generation = 0
        while Generation < MTOs[0].max_gen:
            # For every two generations, conduct one time of knowledge transfer across feature selection tasks
            if Generation % 2 == 0:
                # Build a probabilitics model for each task at the current generation
                Models = [deepcopy(build_model(p[:])) for p in Populations]
            else:
                Models = [n for n in range(len(X))]
            k = 0
            for population, model, hv, mto, mapping in zip(Populations, Models, HVs, MTOs, Mappings):
                t_model = deepcopy(model) # Target model
                s_models = deepcopy(list(set(Models).difference(set([model])))) # Source models
                p, h, tr = mto.run_ga_mto(population, mapping, Generation, t_model, s_models) # Run the algorithm, output hypervolume (h), population (p), transfer coefficient (tr)
                Populations[k] = deepcopy(p)
                HVs[k] += deepcopy([h])
                Trs[k] += deepcopy([tr])
                k += 1
            Generation += 1

        t  = time.time() - start
        
        run += 1
        
        # Save experimental results
        with open(Task_path + '/multi' + '/P' + str(random_seed) + '.pkl', 'wb') as f:
            pickle.dump(Populations, f) 

        with open(Task_path + '/multi' + '/HV' + str(random_seed) + '.pkl', 'wb') as f:
            pickle.dump(HVs, f)

        with open(Task_path + '/multi' + '/TR' + str(random_seed) + '.pkl', 'wb') as f:
            pickle.dump(Trs, f)
            
        with open(Task_path + '/multi' + '/T' + str(random_seed) + '.pkl', 'wb') as f:
            pickle.dump(t, f)        

    
