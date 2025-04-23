#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys

loc = '/projects/hywu/cluster_sims/cluster_finding/data/'
data_loc = loc + 'emulator_data/'

# emu_name = 'fixhod'
# cosmo_id_list_check = np.arange(130, 182, dtype=int)
# hod_id_list_check = np.array([0])

'''
emu_name = 'fixcos'
cosmo_id_list_check = np.array([0])
hod_id_list_check = np.arange(300, 400, dtype=int)
'''

emu_name = 'all'
cosmo_id_list_check = np.arange(130, 182, dtype=int)
hod_id_list_check = np.arange(1000, 1520, dtype=int)


cosmo_id_list_check = cosmo_id_list_check.astype(int)
hod_id_list_check = hod_id_list_check.astype(int)

iz = int(sys.argv[1])
zid = 3+iz
train_loc = loc + f'emulator_train/{emu_name}/train/z0p{zid}00/'
if os.path.isdir(train_loc) == False:
    os.makedirs(train_loc)

#### data vector specs ####
abun_or_lam = 'lam' # 'abun' #
rich_name = 'q180'
phase = 0

#### first, check the hod_id that exists
cosmo_id_list = []
hod_id_list = []


x, x, Nc_target = np.loadtxt('/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/z0p300/model_hod000000/obs_q180_desy1/abundance.dat', unpack=True) # TODO: change to DES counts

for cosmo_id in cosmo_id_list_check:
    #cosmo_para = get_cosmo_para(cosmo_id)
    for hod_id in hod_id_list_check:
        model_name = f'hod{hod_id:0>6d}'
        out_path = data_loc + f'base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/model_{model_name}/'
        survey = 'desy1'
        obs_path = f'{out_path}/obs_{rich_name}_{survey}/'

        ### Check abundance 
        abun_fname = obs_path+'abundance.dat'
        if os.path.exists(abun_fname):

            x, x, Nc = np.loadtxt(abun_fname, unpack=True)
            Nc_ratio = np.mean(Nc / Nc_target)
        
            if Nc_ratio < 0.25 or Nc_ratio > 4:
                pass #print('unrealistic Nc, stop this model')
            else:
                print('reasonable Nc')

                lens_fname = f'{obs_path}/DS_phys_noh_{abun_or_lam}_bin_3.dat'
                if os.path.exists(lens_fname):

                    cosmo_id_list.append(cosmo_id)
                    hod_id_list.append(hod_id)

data = np.array([cosmo_id_list, hod_id_list]).transpose()
np.savetxt(train_loc+'models_done.dat', data, fmt='%-12i', header='cosmo_id, hod_id')
