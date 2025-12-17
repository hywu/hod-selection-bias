#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import fitsio

# make the fake data vector
# !!!Note!!!  Counts are for desy1 area. Counts not scaled by survey area!  For emulator purpuse
# area scaling is done when running MCMC

observation = 'abacus_summit' # 'desy1thre' 'flamingo'
rich_name = 'q180_bg_miscen_R1' #'d30_bg_miscen' #'q180_bg_miscen'


lam = [20, 30, 45, 60, 1000]

nlam = len(lam) - 1

z_data = [0.2, 0.35, 0.5, 0.65]
z_sim = [0.3, 0.4, 0.5]
iz = 0

#binning = 'lam'
# For fiducial data vectors ('flamingo' and 'abacus_summit'), 'abun' and 'lam' are exactly the same.  
# There is only lam/abun distinction in the training set. 
# For mis-spec data vectors ('abcus_summit' only), lam and abun vectors are different (not so clean, but I don't want to regenerate training sets for abun!)

#for observation in ['abacus_summit']: #'flamingo', 

for binning in ['abun', 'lam']:

    redshift = z_sim[iz]

    if observation == 'abacus_summit':
        if redshift == 0.3: z_str = '0p300'
        if redshift == 0.4: z_str = '0p400'
        if redshift == 0.5: z_str = '0p500'
        
        data_loc = '/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/'
        data_loc += f'z{z_str}/model_hod000000/obs_{rich_name}_{observation}/'
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
        #data_loc += f'obs_{rich_name}_desy1/'
        data_loc += f'obs_{rich_name}_{observation}/'
        #print(data_loc)
        ngal = 1e-2

    rp_list = np.logspace(np.log10(0.03), np.log10(30), 15+1)
    rpmin_list = rp_list[:-1]
    rpmax_list = rp_list[1:]
    rpmid_list = np.sqrt(rpmin_list*rpmax_list)
    rp_rad = rpmid_list[rpmid_list>0.2]
    #print('len(rp_rad) = ', len(rp_rad))

    out_loc = f'data_vector/{observation}_{binning}'

    if os.path.isdir(out_loc) == False: 
        os.makedirs(out_loc)

    #### lesing data ####
    DS_data = []
    for ilam in range(nlam):
        rp_in, DS_in = np.loadtxt(data_loc + f'DS_phys_noh_{binning}_bin_{ilam}.dat', unpack=True)
        DS_interp = interp1d(np.log(rp_in), np.log(DS_in))
        DS_data.extend(np.exp(DS_interp(np.log(rp_rad))))
    DS_data = np.array(DS_data)
    np.savetxt(f'{out_loc}/lensing_{rich_name}_z{redshift}.dat', DS_data)
    
    #### counts data (4 bins) ####
    x, x, NC_data = np.loadtxt(data_loc+'abundance.dat',unpack=True)
    np.savetxt(f'{out_loc}/counts_{rich_name}_z{redshift}.dat', NC_data)

    #### cumulative cluster counts (4 thresholds) ####
    #print(NC_data)
    cumNC = np.cumsum(NC_data[::-1])
    #print(cumNC)
    cumNC = cumNC[::-1]
    #print(cumNC)
    np.savetxt(f'{out_loc}/cum_counts_{rich_name}_z{redshift}.dat', cumNC, fmt='%g')

    #### galaxy density ####
    ngal = np.atleast_1d(ngal)
    np.savetxt(f'{out_loc}/ngal_z{redshift}.dat', ngal)


