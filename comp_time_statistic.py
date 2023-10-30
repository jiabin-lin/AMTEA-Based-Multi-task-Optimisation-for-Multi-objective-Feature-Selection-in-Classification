# Statistical analyses on the computational time obtained on the traing datasets

from scipy.stats import wilcoxon
import os 
import pickle
import numpy as np
import pandas as pd


from copy import deepcopy


def sta_test(HVT_1,HVT_2):
    # res = 'non'
    if not np.all(HVT_1 - HVT_2 == 0):
        stat, p = wilcoxon(HVT_1, HVT_2)
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p >= alpha:
            res = 'Same'
            print('Same distribution (fail to reject H0)') 
        else:
            res = 'Different'
            print('Different distribution (reject H0)')
    else:
            print('Same distribution (fail to reject H0)')
            res = 'Same'        
    return res
            
current_path = os.getcwd()

f = open(current_path + '/seeds-30.txt', 'r')
lines = f.readlines()
random_seeds = [int(line) for line in lines]

S_name = 'waveform'

with open(current_path + '/'+ S_name +'/source/T' + str(random_seeds[0]) + '.pkl', 'rb') as f:
    a = pickle.load(f)


T = []
for r in random_seeds:
    with open(current_path + '/'+ S_name +'/source/T' + str(r) + '.pkl', 'rb') as f:
        t = pickle.load(f)
    for i in range(len(a)):
        T += deepcopy([sum(t)])

T_spea2 = deepcopy(T)


T = []
for r in random_seeds:
    with open(current_path + '/'+ S_name +'/multi/T' + str(r) + '.pkl', 'rb') as f:
        t = pickle.load(f)
    for i in range(len(a)):
        T += deepcopy([t])

T_spea2_t = deepcopy(T)


T_spea2 = deepcopy(np.array(T_spea2).T)
T_spea2_t = deepcopy(np.array(T_spea2_t).T)
    
m_T_spea2 = np.mean(T_spea2, axis=0)
m_T_spea2_t = np.mean(T_spea2_t, axis=0)

s_T_spea2 = np.std(T_spea2, axis=0)
s_T_spea2_t = np.std(T_spea2_t, axis=0)


print('Mean of SPEA2: ', m_T_spea2)
print('Mean of SPEA2-AMTEA: ', m_T_spea2_t)

res_spea2 = sta_test(T_spea2,T_spea2_t)


index = ['Problem']


a = np.round(m_T_spea2,3)
b = np.round(s_T_spea2,3)
c = [str(a)+'('+str(b)+')']

# for i, j in zip(a, b):
#     c += [str(i)+'('+str(j)+')']
mh_spea2 = deepcopy(c)

a = np.round(m_T_spea2_t,3)
b = np.round(s_T_spea2_t,3)
c = []
c = [str(a)+'('+str(b)+')']
# for i, j in zip(a, b):
#     c += [str(i)+'('+str(j)+')']
mh_spea2_t = deepcopy(c)


d = {'SPEA2-FS_mean(std)' : pd.Series(mh_spea2, index),
     'SPEA2-FSMTO_mean(std)' : pd.Series(mh_spea2_t, index),
     'SPEA2-FS-RES' : pd.Series(res_spea2, index)}

df = pd.DataFrame(d)
print(df)





















