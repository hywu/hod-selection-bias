#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from astropy.io import fits
import pandas as pd
import os, sys, glob, argparse
#### my functions ####
sys.path.append('../utils')
from fid_hod import Ngal_S20_poisson
from fid_hod import Ngal_S20_gauss
from draw_sat_position import draw_sat_position


parser = argparse.ArgumentParser()
parser.add_argument('--which_sim', type=str, required=True, help='')
parser.add_argument('--model_id', type=int, required=True, help='')
args = parser.parse_args()

model_list = pd.read_csv('model_list.csv')
model = model_list.iloc[args.model_id]
#print(model)
alpha = model['alpha']
lgM1 = model['lgM1']
kappa = model['kappa']
lgMcut = model['lgMcut']
sigmalogM = model['sigmalogM']
sigma_intr = model['sigma_intr']

if args.which_sim == 'mini_uchuu':
    output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'

out_path = f'{output_loc}/model_{args.model_id}/'
if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')


## save the model to a one-row csv
df_save = pd.DataFrame(columns=list(model.keys()))
df_save.loc[0] = model
df_save.to_csv(f'{out_path}/model.csv', index=False)



## read in halo catalog
redshift = 0.3


# input_loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
# data = h5py.File(input_loc+f'MiniUchuu_halolist_z0p30.h5', 'r')
# hid = np.array(data['id'])
# pid = np.array(data['pid'])
# sel1 = (pid == -1)
# M200m = np.array(data['M200b'])[sel1]
# sel2 = M200m > 1e11
# M200m_all = M200m[sel2]
# gid_all = hid[sel1][sel2]
# x_halo_all = np.array(data['x'])[sel1][sel2]
# y_halo_all = np.array(data['y'])[sel1][sel2]
# z_halo_all = np.array(data['z'])[sel1][sel2]
if args.which_sim == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    rmu = ReadMiniUchuu()
    rmu.read_halos()
    boxsize = rmu.boxsize
    x_halo_all = rmu.xh
    y_halo_all = rmu.yh
    z_halo_all = rmu.zh
    M200m_all = rmu.M200m
    gid_all = rmu.gid

def calc_one_layer(pz_min, pz_max):
    sel = (z_halo_all >= pz_min)&(z_halo_all < pz_max)
    x_halo_sub = x_halo_all[sel]
    y_halo_sub = y_halo_all[sel]
    z_halo_sub = z_halo_all[sel]
    M200m_sub = M200m_all[sel]
    gid_sub = gid_all[sel]
    nhalo = len(x_halo_sub)

    print('nhalo', nhalo)
    hid_out = []
    m_out = []
    x_out = []
    y_out = []
    z_out = []
    iscen_out = []

    for ih in range(nhalo):
        if sigma_intr < 1e-6: # poisson
            Ntot = Ngal_S20_poisson(M200m_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM) 
        else:
            Ntot = Ngal_S20_gauss(M200m_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM, sigma_intr=sigma_intr)

        if Ntot >= (1-1e-6):
            hid_out.append(gid_sub[ih])
            m_out.append(M200m_sub[ih])
            x_out.append(x_halo_sub[ih])
            y_out.append(y_halo_sub[ih])
            z_out.append(z_halo_sub[ih])
            iscen_out.append(1)
            Nsat = Ntot - 1
            if Nsat > 1e-4:
                px, py, pz = draw_sat_position(redshift, M200m_sub[ih], Nsat)
                hid_out.extend(np.zeros(len(px)) + gid_sub[ih])
                m_out.extend(np.zeros(len(px)) + M200m_sub[ih])
                x_out.extend(x_halo_sub[ih] + px)
                y_out.extend(y_halo_sub[ih] + py)
                z_out.extend(z_halo_sub[ih] + pz)
                iscen_out.extend(np.zeros(len(px)))

    x_out = np.array(x_out)
    y_out = np.array(y_out)
    z_out = np.array(z_out)
    x_out[x_out < 0] += boxsize
    y_out[y_out < 0] += boxsize
    z_out[z_out < 0] += boxsize
    x_out[x_out > boxsize] -= boxsize
    y_out[y_out > boxsize] -= boxsize
    z_out[z_out > boxsize] -= boxsize

    data = np.array([hid_out, m_out, x_out, y_out, z_out, iscen_out]).transpose()
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.dat'
    np.savetxt(ofname, data, fmt='%-12i %-15.12e  %-12.12g  %-12.12g  %-12.12g %-12i', header='haloid, M200m, px, py, pz, iscen') # need a few more decimal places


def calc_one_bin(ibin):
    calc_one_layer(pz_min=ibin*40, pz_max=(ibin+1)*40)

def merge_files():
    fname_list = glob.glob(f'{out_path}/temp/gals_*.dat')
    hid_out = []
    m_out = []
    x_out = []
    y_out = []
    z_out = []
    iscen_out = []
    for fname in fname_list:
        data = pd.read_csv(fname, delim_whitespace=True, dtype=np.float64, comment='#', 
                        names=['haloid', 'm', 'px', 'py', 'pz', 'iscen'])
        hid_out.extend(data['haloid'])
        m_out.extend(data['m'])
        x_out.extend(data['px'])
        y_out.extend(data['py'])
        z_out.extend(data['pz'])
        iscen_out.extend(data['iscen'])
        
    hid_out = np.array(hid_out)
    m_out = np.array(m_out)
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    z_out = np.array(z_out)
    iscen_out = np.array(iscen_out)
    sel = np.argsort(-m_out)

    cols=[
      fits.Column(name='haloid', format='K' ,array=hid_out[sel]),
      fits.Column(name='M200m', format='E',array=m_out[sel]),
      fits.Column(name='px', format='D' ,array=x_out[sel]),
      fits.Column(name='py', format='D',array=y_out[sel]),
      fits.Column(name='pz', format='D',array=z_out[sel]),
      fits.Column(name='iscen', format='K',array=iscen_out[sel])
    ]
    coldefs = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    tbhdu.writeto(f'{out_path}/gals.fit', overwrite=True)


if __name__ == '__main__':

    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    
    start = stop
    n_job2 = 10
    p = Pool(n_job2)
    p.map(calc_one_bin, range(n_job2))
    stop = timeit.default_timer()
    print('galaxies took', stop - start, 'seconds')

    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', stop - start, 'seconds')
