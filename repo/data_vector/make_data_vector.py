#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

# make the fake data
# !!!Note!!!  Not scaled by survey area!  For emulator purpuse
# area scaling is done when running MCMC

#observation = 'flamingo' #
#observation = 'abacus_summit' # 'desy1thre'
rich_name = 'q180_bg_miscen'

# if observation == 'desy1thre':
#     lam = [20, 1000]
# else:
lam = [20, 30, 45, 60, 1000]

nlam = len(lam) - 1

z_data = [0.2, 0.35, 0.5, 0.65]
z_sim = [0.3, 0.4, 0.5]
iz = 0

binning = 'lam' # old parameter. should get rid of!
# for binning in ['abun', 'lam']:
# Note! There should be no distinction in binning for data vector!  Only in training set. 

for observation in ['flamingo', 'abacus_summit']:

    redshift = z_sim[iz]

    if observation == 'abacus_summit':
        if redshift == 0.3: z_str = '0p300'
        if redshift == 0.4: z_str = '0p400'
        if redshift == 0.5: z_str = '0p500'
        
        data_loc = '/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/'
        data_loc += f'z{z_str}/model_hod000000/obs_{rich_name}_desy1/' # only ran desy1 with abacus
        # # get the radius
        # rp_list = np.logspace(np.log10(0.03), np.log10(30), 15+1)
        # rpmin_list = rp_list[:-1]
        # rpmax_list = rp_list[1:]
        # rpmid_list = np.sqrt(rpmin_list*rpmax_list)
        # rp_rad = rpmid_list[rpmid_list>0.2]
        ngal = np.loadtxt(data_loc+'../gal_density.dat')
    
    if observation == 'flamingo':
        model_name = 'redmagic_chi6_1e-02_HBT' #'redmagic_chi2_6e-03_HBT'
        data_loc = '/projects/hywu/cluster_sims/cluster_finding/data/flamingo/output_L1000N3600/HYDRO_FIDUCIAL/'
        data_loc += f'z{redshift}/model_{model_name}/' 
        data_loc += f'obs_{rich_name}_desy1/'
        #print(data_loc)
        ngal = 1e-2

    rp_list = np.logspace(np.log10(0.03), np.log10(30), 15+1)
    rpmin_list = rp_list[:-1]
    rpmax_list = rp_list[1:]
    rpmid_list = np.sqrt(rpmin_list*rpmax_list)
    rp_rad = rpmid_list[rpmid_list>0.2]
    #print('len(rp_rad) = ', len(rp_rad))

    if os.path.isdir(f'data_vector_{observation}') == False: 
        os.makedirs(f'data_vector_{observation}')

    #### lesing data ####
    DS_data = []
    for ilam in range(nlam):
        rp_in, DS_in = np.loadtxt(data_loc + f'DS_phys_noh_{binning}_bin_{ilam}.dat', unpack=True)
        DS_interp = interp1d(np.log(rp_in), np.log(DS_in))
        DS_data.extend(np.exp(DS_interp(np.log(rp_rad))))
    DS_data = np.array(DS_data)
    np.savetxt(f'data_vector_{observation}/lensing_{rich_name}_z{redshift}.dat', DS_data)
    
    #### counts data ####
    x, x, NC_data = np.loadtxt(data_loc+'abundance.dat',unpack=True)
    np.savetxt(f'data_vector_{observation}/counts_{rich_name}_z{redshift}.dat', NC_data)

    #### counts ####
    ngal = np.atleast_1d(ngal)
    np.savetxt(f'data_vector_{observation}/ngal_z{redshift}.dat', ngal)

    #### cumulative density ####
