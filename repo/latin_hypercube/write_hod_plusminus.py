#!/usr/bin/env python
import numpy as np
import pandas as pd
import os

#nsample = 100

# sampler = qmc.LatinHypercube(d=ndim)
# sample_list = sampler.random(n=nsample)

alpha_fid = 1
# alpha_max = 1.5
# alpha_min = 0.5
# alpha_range = alpha_max - alpha_min 

lgM1_fid = 12.9
# lgM1_min = 12
# lgM1_max = 13.5
# lgM1_range = lgM1_max - lgM1_min

lgkappa_fid = 0
# lgkappa_min = -0.5
# lgkappa_max = 0.5
# lgkappa_range = lgkappa_max - lgkappa_min

lgMcut_fid = 11.7
# lgMcut_min = 11
# lgMcut_max = 12.5
# lgMcut_range = lgMcut_max - lgMcut_min

sigmalogM_fid = 0.1
# sigmalogM_min = 0.05
# sigmalogM_max = 0.2
# sigmalogM_range = sigmalogM_max - sigmalogM_min

sigmaintr_fid = 0

#miscen
f_miscen_fid = 0.165
# f_miscen_min = 0.1
# f_miscen_max = 0.2

tau_miscen_fid = 0.166
# tau_miscen_min = 0.1
# tau_miscen_max = 0.2

depth_fid = 180

A_fid = 1

B_fid = 0

### hod_id starting from 1 (0 is the fidicial)
npara = 11
isample = 1
for ipara in range(npara):
    for p_or_m in [1, -1]:
        pm = np.zeros(npara)
        pm[ipara] = p_or_m


        if ipara == 4 and pm[ipara] == -1: # Take care of sigmalogM (don't use 0)
            pm[ipara] = 2

        if ipara == 5 and pm[ipara] == -1: # Take care of sigmaintr (don't use -0.1)
            pm[ipara] = 2

        new_row = {
        'hod_id': 0+isample, 
        'alpha': alpha_fid + pm[0]*0.1,
        'lgM1': lgM1_fid + pm[1]*0.1,
        'lgkappa': lgkappa_fid + pm[2]*0.1,
        'lgMcut': lgMcut_fid + pm[3]*0.1,
        'sigmalogM': sigmalogM_fid + pm[4]*0.1, 
        'sigmaintr': sigmaintr_fid + pm[5]*0.1, #fid is 0 
        'f_miscen':f_miscen_fid + pm[6]*0.05,
        'tau_miscen':tau_miscen_fid + pm[7]*0.05,
        'depth': depth_fid + pm[8]*30,
        'A': A_fid + pm[9]  * 0.1,
        'B': B_fid + pm[10] * 0.1
        }
        
        new_row_df = pd.DataFrame([new_row])
        filename = 'parameters/hod_rich_AB_plusminus.csv'
        new_row_df.to_csv(filename, float_format='%-4g ', mode='a', header=not os.path.isfile(filename), index=False)
        isample += 1
        