#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from scipy.interpolate import CubicSpline
from hod.utils.get_para_abacus_summit import get_cosmo_para, get_hod_para

loc = '/projects/hywu/cluster_sims/cluster_finding/data/'
data_loc = loc + 'emulator_data/'

emu_name = sys.argv[1]
binning = sys.argv[2]  #'AB' # 'lam' # 'abun'
survey = sys.argv[3] #'desy1thre' # 'desy1'
iz = 0

zid = 3+iz
train_loc = loc + f'emulator_train/{emu_name}/z0p{zid}00/{survey}_{binning}/'

#### data vector specs ####
emu_name = 'rad'
rich_name = 'q180_bg_miscen' #'q180_miscen'
phase = 0

cosmo_id_list, hod_id_list = np.loadtxt(train_loc+'models_done.dat', unpack=True, dtype=int)

n_models = len(hod_id_list)
print('n_models', n_models)

filename = f'{train_loc}/parameters.csv'
try:
    os.remove(filename)
except OSError:
    pass

outfile_all = open(f'{train_loc}/parameters_all.dat', 'w')
outfile_all.write('# master_id, sigma8, Om0, ns, Ob0, w0, wa, Nur, alpha_s, ')
outfile_all.write('alpha, lgM1, lgkappa, lgMcut, sigmalogM \n')

#for cosmo_id in cosmo_id_list:
#    for hod_id in hod_id_list:
for imodel in range(n_models):
    cosmo_id = cosmo_id_list[imodel]
    hod_id = hod_id_list[imodel]
    cosmo_para = get_cosmo_para(cosmo_id)

    hod_para = get_hod_para(hod_id)
    para = cosmo_para | hod_para

    new_row_df = pd.DataFrame([para])
    new_row_df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False)

    outfile_all.write('%2i %12g %12g %12g %12g %8g %8g %8g %8g '%(para['hod_id'], para['sigma8'], para['OmegaM'], para['ns'], para['OmegaB'], para['w0'], para['wa'], para['Nur'], para['alpha_s']))

    outfile_all.write('%12g %12g %12g %12g %12g\n'%(para['alpha'], para['lgM1'], para['lgkappa'], para['lgMcut'], para['sigmalogM'])) #, para['sigmaintr']

#### create the raining set (pre-PCA) ####
#### no train-test split.  Use LOOE later
#### all units are phys, no-h.  

### training set for pca: many radial bins
### training set for bin: DES radial bins

rp_master_pca = np.logspace(-2,2,100)
#rp_master_rad, DS, DS_err = np.loadtxt(f'../../hod/y1/data/y1_DS_bin_z_0.2_0.35_lam_20_30.dat', unpack=True)

# rp_master_rad = np.logspace(np.log10(0.03), np.log10(30), 15)
# rp_master_rad = rp_master_rad[rp_master_rad>0.2]
rp_list = np.logspace(np.log10(0.03), np.log10(30), 15+1)
rpmin_list = rp_list[:-1]
rpmax_list = rp_list[1:]
rpmid_list = np.sqrt(rpmin_list*rpmax_list)
rp_master_rad = rpmid_list[rpmid_list>0.2]
print('len(rp_rad) = ', len(rp_master_rad))

if survey == 'desy1':
    nlam = 4
if survey == 'desy1thre':
    nlam = 1
    
for ilam in range(nlam):
    outfile_pca = open(f'{train_loc}/DS_{binning}_bin_{ilam}_pca.dat', 'w')
    outfile_rad = open(f'{train_loc}/DS_{binning}_bin_{ilam}_rad.dat', 'w')

    outfile_pca.write('#log DS (phys, no h) \n')
    outfile_rad.write('#log DS (phys, no h) \n')

    for imodel in range(n_models):
        cosmo_id = cosmo_id_list[imodel]
        hod_id = hod_id_list[imodel]

        model_name = f'hod{hod_id:0>6d}'
        out_path = data_loc + f'base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/model_{model_name}/'
        lens_fname = f'{out_path}/obs_{rich_name}_{survey}/DS_phys_noh_{binning}_bin_{ilam}.dat'
        rp, DS = np.loadtxt(lens_fname, unpack=True)
        x = np.log(rp)
        y = np.log(DS)
        y = np.nan_to_num(y, nan=0) ## TODO
        spl = CubicSpline(x, y)
        y_smooth_pca = spl(np.log(rp_master_pca))
        y_smooth_rad = spl(np.log(rp_master_rad))

        for x in y_smooth_pca:
            outfile_pca.write('%g \t'%x)
        outfile_pca.write('\n')

        for x in y_smooth_rad:
            outfile_rad.write('%g \t'%x)
        outfile_rad.write('\n')

    outfile_pca.close()
    outfile_rad.close()

# save the radius #
np.savetxt(f'{train_loc}/rp_pca.dat', rp_master_pca, header='rp (pMpc, noh)')
np.savetxt(f'{train_loc}/rp_rad.dat', rp_master_rad, header='rp (pMpc, noh)')


###### abundance ######
outfile_abun = open(f'{train_loc}/abundance.dat', 'w')
outfile_abun.write('# ln(counts)\n')
for imodel in range(n_models):
    cosmo_id = cosmo_id_list[imodel]
    hod_id = hod_id_list[imodel]

    model_name = f'hod{hod_id:0>6d}'
    out_path = data_loc + f'base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/model_{model_name}/'
    abun_fname = f'{out_path}/obs_{rich_name}_desy1/abundance.dat'
    lam_min, lam_max, abun = np.loadtxt(abun_fname, unpack=True)
    for x in abun:
        outfile_abun.write('%12g \t'%np.log(x))
    outfile_abun.write('\n')
outfile_abun.close()

