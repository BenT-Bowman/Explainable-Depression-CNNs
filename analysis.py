import numpy as np
from scipy.stats import ttest_ind, ttest_rel


l1 = np.array([0.9446916719643992, 0.7686274509803922, 0.5782975958414555, 0.5309043591411842, 0.5353185595567868, 0.8126315789473684]) # EEGNet
l2 = np.array([0.9198982835346472, 0.8339869281045752, 0.8641975308641975, 0.8158750813272609, 0.9182825484764543, 0.8147368421052632]) # EEGNet CAEW
l3 = np.array([0.9682136045772409, 0.807843137254902, 0.8973359324236517, 0.808067664281067, 0.9452908587257618, 0.8315789473684211]) # DeprNet

print("mean: ", l1.mean(), l2.mean(), l3.mean())
print("std div: ", l1.std(), l2.std(), l3.std())
t_stat_eegnet, p_value_eegnet = ttest_ind(l1, l3)
print(f"{t_stat_eegnet=}, {p_value_eegnet=}")