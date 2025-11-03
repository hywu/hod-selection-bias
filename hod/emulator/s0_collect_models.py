#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys

loc = '/projects/hywu/cluster_sims/cluster_finding/data/'
data_loc = loc + 'emulator_data/'

emu_name = sys.argv[1]
binning = sys.argv[2]  #'AB' # 'lam' # 'abun'
observation = sys.argv[3]  #'desy1thre' # 'desy1'
rich_name = 'q180_bg_miscen' #'q180_miscen'
iz = 0
phase = 0

'''
if emu_name == 'fixhod':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.array([0])

if emu_name == 'fixcos':
    cosmo_id_list_check = np.array([0])
    hod_id_list_check = np.arange(300, 400, dtype=int)

if emu_name == 'narrow_miscen':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.arange(1000, 1520, dtype=int)

if emu_name == 'narrow':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.arange(1000, 2000, dtype=int)

if emu_name == 'wide':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.arange(2000, 2520, dtype=int)


if emu_name == 'iter1':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.arange(1000, 3100, dtype=int)
'''

if emu_name == 'all':
    cosmo_id_list_check = np.arange(130, 182, dtype=int)
    hod_id_list_check = np.arange(1000, 5000, dtype=int)


cosmo_id_list_check = cosmo_id_list_check.astype(int)
hod_id_list_check = hod_id_list_check.astype(int)

zid = 3+iz
train_loc = loc + f'emulator_train/z0p{zid}00_{emu_name}/{observation}_{binning}/'

if os.path.isdir(train_loc) == False:
    os.makedirs(train_loc)

# collect the models with lensing calculation finished
cosmo_id_list = []
hod_id_list = []

for cosmo_id in cosmo_id_list_check:
    for hod_id in hod_id_list_check:
        model_name = f'hod{hod_id:0>6d}'
        out_path = data_loc + f'base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/model_{model_name}/'


        obs_path = f'{out_path}/obs_{rich_name}_{observation}/'

        if observation in ['desy1', 'flamingo', 'abacus_summit']:
            lens_fname = f'{obs_path}/DS_phys_noh_{binning}_bin_3.dat'
        if observation == 'desy1thre':
            lens_fname = f'{obs_path}/DS_phys_noh_{binning}_bin_0.dat'

        if os.path.exists(lens_fname):
            cosmo_id_list.append(cosmo_id)
            hod_id_list.append(hod_id)

data = np.array([cosmo_id_list, hod_id_list]).transpose()
np.savetxt(train_loc+'models_done.dat', data, fmt='%-12i', header='cosmo_id, hod_id')
print('file saved', train_loc+'models_done.dat')