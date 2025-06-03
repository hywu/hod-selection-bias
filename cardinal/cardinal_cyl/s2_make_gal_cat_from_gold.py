#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import timeit
import os, sys
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.286)
z_list = np.arange(0,5,0.0001)
chi_list = cosmo.comoving_distance(z_list).value
redshift_chi_interp = interp1d(chi_list, z_list)

from magnitude_cut import mag_i_lim_Rykoff14
from member_color_interp import *

chisq_cut = int(sys.argv[1])

output_loc = '/projects/hywu/cluster_sims/cluster_finding/data/cardinal_cyl/'
output_loc += f'model_chisq{chisq_cut}/'

if os.path.isdir(output_loc+'temp/') == False: 
    os.makedirs(output_loc+'temp/')


card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
fname = card_loc + 'Cardinal-3_v2.0_Y6a_gold.h5'
f = h5py.File(fname, 'r')
data = f['catalog/gold']

px_all = data['px'][:]
py_all = data['py'][:]
pz_all = data['pz'][:]
ra_all = data['ra'][:]
dec_all = data['dec'][:]
chi_all = np.sqrt(px_all**2 + py_all**2 + pz_all**2) # there is no redshift. sigh.
mag_g_all = data['mag_g'][:]
mag_r_all = data['mag_r'][:]
mag_i_all = data['mag_i'][:]
mag_z_all = data['mag_z'][:]
z_all = redshift_chi_interp(chi_all)

'''
# for quick debugging. do not delete
px_all = np.zeros(1000)
py_all = np.zeros(1000)
pz_all = np.zeros(1000)
ra_all = np.zeros(1000)
dec_all = np.zeros(1000)
mag_g_all = np.zeros(1000)+20
mag_r_all = np.zeros(1000)+18
mag_i_all = np.zeros(1000)+18
mag_z_all = np.zeros(1000)+16
chi_all = np.zeros(1000) + 520
z_all =  np.zeros(1000) + 520
'''
print('finished reading gold')

dz = 0.01
#zmin_list = np.arange(0.2, 0.641, dz)
zmin_list = np.arange(0.15, 0.75, dz)

def calc_one_bin(iz):
    zmin = zmin_list[iz]
    zmax = zmin + dz
    zmid = zmin + 0.5 *dz

    

    '''
    mag_i_cut = mag_i_lim_Rykoff14(zmid)
    print('mag_i_cut', mag_i_cut)

    g_r_mean = g_r_vs_redshift(zmid)
    g_r_std = sigma_g_r_vs_redshift(zmid)

    r_i_mean = r_i_vs_redshift(zmid)
    r_i_std = sigma_r_i_vs_redshift(zmid)

    i_z_mean = i_z_vs_redshift(zmid)
    i_z_std = sigma_i_z_vs_redshift(zmid)
    '''
    sel = (z_all > zmin)&(z_all < zmax)

    chi_gal = chi_all[sel]
    ra_gal = ra_all[sel] # deg
    dec_gal = dec_all[sel] 

    g_r = mag_g_all[sel] - mag_r_all[sel]
    r_i = mag_r_all[sel] - mag_i_all[sel]
    i_z = mag_i_all[sel] - mag_z_all[sel]

    chisq = (g_r - g_r_mean)**2 / g_r_std**2
    chisq += (r_i - r_i_mean)**2 / r_i_std**2
    chisq += (i_z - i_z_mean)**2 / i_z_std**2

    sel2 = (chisq < chisq_cut)

    chi_gal = chi_gal[sel2]
    ra_gal = ra_gal[sel2] # deg
    dec_gal = dec_gal[sel2] # deg

    data = np.array([ra_gal, dec_gal, chi_gal]).transpose()
    ofname = output_loc + f'temp/gals_{iz}.dat'
    #with open(ofname, "ab") as f:
    np.savetxt(ofname, data, fmt='%12g %12g %12g', header='ra dec chi'
            , comments='') # without pound sign, without comma!


if __name__ == "__main__":
    n_parallel = len(zmin_list)

    #calc_one_bin(0)
    
    import os
    from concurrent.futures import ProcessPoolExecutor
    n_cpu = os.getenv('SLURM_CPUS_PER_TASK')

    if n_cpu is not None:
        n_cpu = int(n_cpu)
        print(f'Assigned CPUs: {n_cpu}') 
    else:
        print('Not running under SLURM or the variable is not set.') 
        n_cpu = 1

    n_workers = int(max(1, n_cpu*0.8))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(calc_one_bin, range(n_parallel)):
            if result: print(result)  # output error
    stop = timeit.default_timer()
    
    from hod.utils.merge_files import merge_files
    merge_files(in_fname=f'{output_loc}/temp/gals_*.dat', 
        out_fname=f'{output_loc}/gals.fit', 
        nfiles_expected=n_parallel, delete=False)
    