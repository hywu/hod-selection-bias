#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import pandas as pd
import h5py
from multiprocessing import Pool
from astropy.io import fits
import pandas as pd
import os, sys, glob 
import yaml
#### my functions ####
sys.path.append('../utils')
from fid_hod import Ngal_S20_poisson
#from fid_hod import Ngal_S20_gauss
from draw_sat_position import DrawSatPosition
#from read_yml import ReadYML


#### read in the yaml file  ####
yml_fname = sys.argv[1]
dsp = DrawSatPosition(yml_fname)
#./make_gal_cat.py yml/mini_uchuu_fid_hod.yml
#./make_gal_cat.py yml/abacus_summit_fid_hod.yml

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


output_loc = para['output_loc']
model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}/'
redshift = para['redshift']
alpha = para['alpha']
lgM1 = para['lgM1']
lgkappa = para['lgkappa']
lgMcut = para['lgMcut']
sigmalogM = para['sigmalogM']
sigma_intr = para['sigmaintr']
kappa = 10**lgkappa

Mmin = para.get('Mmin', 1e11)


if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')


with open(f'{out_path}/para.yml', 'w') as outfile:
    yaml.dump(para, outfile, default_flow_style=False)

print('output is at ' + out_path)

if para['nbody'] == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    readcat = ReadMiniUchuu(para['nbody_loc'])
    readcat.read_halos(Mmin)
    boxsize = readcat.boxsize
    x_halo_all = readcat.xh
    y_halo_all = readcat.yh
    z_halo_all = readcat.zh
    mass_all = readcat.M200m
    hid_all = readcat.hid

if para['nbody'] == 'abacus_summit':
    from read_abacus_summit import ReadAbacusSummit
    readcat = ReadAbacusSummit(para['nbody_loc'])
    readcat.read_halos(Mmin)
    boxsize = readcat.boxsize
    x_halo_all = readcat.xh
    y_halo_all = readcat.yh
    z_halo_all = readcat.zh
    mass_all = readcat.mass # sigh, this is actually Mvir
    hid_all = readcat.hid



def calc_one_layer(pz_min, pz_max):
    sel = (z_halo_all >= pz_min)&(z_halo_all < pz_max)
    x_halo_sub = x_halo_all[sel]
    y_halo_sub = y_halo_all[sel]
    z_halo_sub = z_halo_all[sel]
    mass_sub = mass_all[sel]
    hid_sub = hid_all[sel]
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
            Ncen, Nsat = Ngal_S20_poisson(mass_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM) 
        else:
            Ncen, Nsat = Ngal_S20_gauss(mass_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM, sigma_intr=sigma_intr)

        if Ncen > 0.5:
            hid_out.append(hid_sub[ih])
            m_out.append(mass_sub[ih])
            x_out.append(x_halo_sub[ih])
            y_out.append(y_halo_sub[ih])
            z_out.append(z_halo_sub[ih])
            iscen_out.append(1)
            #Nsat = Ntot - 1
            if Nsat > 0.5:
                px, py, pz = dsp.draw_sat_position(mass_sub[ih], Nsat)
                hid_out.extend(np.zeros(len(px)) + hid_sub[ih])
                m_out.extend(np.zeros(len(px)) + mass_sub[ih])
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
    np.savetxt(ofname, data, fmt='%-12i %-15.12e %-12.12g %-12.12g %-12.12g %-12i', header='haloid, M200m, px, py, pz, iscen') # need a few more decimal places


n_parallel = 100
n_layer = boxsize / n_parallel

def calc_one_bin(ibin):
    calc_one_layer(pz_min=ibin*n_layer, pz_max=(ibin+1)*n_layer)

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

    os.system(f'rm -rf {out_path}/temp/gals_*.dat')


if __name__ == '__main__':

    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    
    start = stop
    p = Pool(n_parallel)
    p.map(calc_one_bin, range(n_parallel))
    stop = timeit.default_timer()
    print('galaxies took', stop - start, 'seconds')

    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', stop - start, 'seconds')
