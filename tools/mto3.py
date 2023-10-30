"""
The class of multi-task SPEA2 feature selection
"""

import random
import math

from copy import deepcopy
import numpy as np

from numpy import mean
# import time


from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# from tools.Base_functions import build_model

from deap import creator, tools, base, algorithms
# import deap

from deap.tools.emo import sortNondominated
from deap.benchmarks.tools import hypervolume

from tools.ProbabilisticModel import ProbabilisticModel, MixtureModel

class MTO(object):
    
    def __init__(self, X, Y, random_seed, creator):
 
        
        self.features = X.columns
        self.pop_size = math.ceil(len(self.features))
        if self.pop_size > 200:
            self.pop_size = 200
    
        self.max_gen = 5*self.pop_size
    
        if self.max_gen > 200:
            self.max_gen = 200
        self.random_seed = random_seed
        self.reps = 1        
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("bit", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.bit, n = len(self.features))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual, n = self.pop_size)
        self.toolbox.register("mate", tools.cxOnePoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb = float(1/len(self.features)))
        self.toolbox.register("select", tools.selSPEA2)
        self.toolbox.register("evaluate", self.evalFitness, X = X, y = Y)
        
        self.NGEN = self.max_gen
        self.MU = self.pop_size


    def evalFitness(self, individual, X, y):
        model = tree.DecisionTreeClassifier(criterion="entropy",random_state=self.random_seed)
        if sum(individual) == 0:
            # print("Empty dataset")
            return len(self.features)+2, 1+1
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=self.random_seed)
            scores = cross_val_score(model, X[self.features[list(map(bool,individual))]], y, scoring='accuracy', cv=cv, n_jobs=-1)
            return sum(individual)/len(self.features), 1-mean(scores)
    
    def evalFitness_final(self, individual, X_train, X_test, y_train, y_test):
        model = tree.DecisionTreeClassifier(criterion="entropy",random_state=self.random_seed)
        if sum(individual) == 0:
            # print("Empty dataset")
            return len(self.features)+2, 1+1
        else:
            model.fit(X_train[self.features[list(map(bool,individual))]], y_train)
            return sum(individual)/len(self.features), 1-model.score(X_test[self.features[list(map(bool,individual))]], y_test)


    def run_ga_mto(self, Population, Mapping, Generation, T_model, S_models):

            population = deepcopy(Population)

            if Generation % 2 == 0:
                
                t_model = deepcopy(T_model)
                s_models = deepcopy(S_models)
                
                # Transform the source probabilistics models to match the target task
                for m in range(len(s_models)):
                    s_new = deepcopy(t_model)
                    s_new.probOne = s_new.probOne * 0
                    s_new.probZero = 1 - s_new.probOne
                    s_new.probOne_noisy = s_new.probOne_noisy * 0
                    s_new.probZero_noisy = 1 - s_new.probOne_noisy
                    for u in range(len(Mapping[0][0])):
                        # print('u: ', u)
                        s_new.probOne[Mapping[m][0][u]] = deepcopy(s_models[m].probOne[Mapping[m][1][u]])
                        s_new.probZero[Mapping[m][0][u]] = deepcopy(s_models[m].probZero[Mapping[m][1][u]])
                        s_new.probOne_noisy[Mapping[m][0][u]] = deepcopy(s_models[m].probOne_noisy[Mapping[m][1][u]])
                        s_new.probZero_noisy[Mapping[m][0][u]] = deepcopy(s_models[m].probZero_noisy[Mapping[m][1][u]])
                    s_models[m] = deepcopy(s_new) 
                
            
            if Generation % 2 == 0:
                # Transfer knowledge
                # The new solutions to the target task are generated based on a mixture probabilistic model.
                M = np.zeros((len(population), len((population[0]))))
                for i in range(len(population)):
                    M[i,:] = population[i]
                mixModel = MixtureModel(s_models)
                mixModel.createTable(M, True, 'umd')
                mixModel.EMstacking()
                mixModel.mutate()
                transfer_coefficient = mixModel.alpha
                offspring = self.toolbox.population()
    
                for i in range(self.pop_size):
                    offspring[i] = creator.Individual(mixModel.sample(1)[0].astype(int).tolist())
            else:
                # Not transfer knowledge
                # New solutions are generated based on genetic operators such as crossover and mutation.
                transfer_coefficient = np.zeros(len(Mapping)+1)
                # random.seed(r)
                offspring = algorithms.varOr(population, self.toolbox, lambda_ = self.pop_size, cxpb = 0.5, mutpb = 0.1)    
            # Evaluate the new solutions
            fits = self.toolbox.map(self.toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            # Select good solutions based on SPEA2.
            population = self.toolbox.select(population + offspring, k = self.pop_size)
            # Calculate hypervolume
            hv = deepcopy(hypervolume(sortNondominated(population,k=self.pop_size)[0],[1.1,1.1]))
        
            return population, hv, transfer_coefficient

