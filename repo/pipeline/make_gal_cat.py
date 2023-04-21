#!/usr/bin/env python
import timeit
start = timeit.default_timer()
start_master = start * 1
import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
import pandas as pd
import os
import sys
import glob
import yaml
from concurrent.futures import ProcessPoolExecutor

#### my functions ####
sys.path.append('../utils')
from fid_hod import Ngal_S20_poisson

#### read in the yaml file  ####
yml_fname = sys.argv[1]
#./make_gal_cat.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

"TODO: parallel version doesn't output error message"

output_loc = para['output_loc']
model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}/'
redshift = para['redshift']
alpha = para['alpha']
lgkappa = para['lgkappa']
lgMcut = para['lgMcut']
sigmalogM = para['sigmalogM']
sigma_intr = para['sigmaintr']
kappa = 10**lgkappa
lgM20 = para.get('lgM20', None)
if lgM20 == None:
    lgM1 = para['lgM1']
else:
    M1 = 20**(-1/alpha) *(10**lgM20 - 10**lgkappa * 10**lgMcut)
    lgM1 = np.log10(M1)

pec_vel = para.get('pec_vel', False)

seed = para.get('seed', 42)
np.random.seed(seed=seed) # for scipy

Mmin = para.get('Mmin', 1e11)

sat_from_part = para.get('sat_from_part', False)
if sat_from_part == True:
    print('drawing sat from particles')
    #from draw_sat_positions_from_particles import DrawSatPositionsFromParticles
    #dsp_part = DrawSatPositionsFromParticles(yml_fname)
    from draw_sat_positions_from_particles_layer import DrawSatPositionsFromParticlesLayer
    dsp_part = DrawSatPositionsFromParticlesLayer(yml_fname)
    Mmin_part = 10**12.5 # only draw particles for halos above this mass

from draw_sat_position import DrawSatPosition
dsp_mc = DrawSatPosition(yml_fname)

if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')

with open(f'{out_path}/para.yml', 'w') as outfile:
    yaml.dump(para, outfile, default_flow_style=False)

print('output is at ' + out_path)

if para['nbody'] == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    readcat = ReadMiniUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'uchuu':
    from read_uchuu import ReadUchuu
    readcat = ReadUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'abacus_summit':
    from read_abacus_summit import ReadAbacusSummit
    readcat = ReadAbacusSummit(para['nbody_loc'], redshift)

if para['nbody'] == 'tng_dmo':
    from read_tng_dmo import ReadTNGDMO
    halofinder = para.get('halofinder', 'rockstar')
    readcat = ReadTNGDMO(para['nbody_loc'], halofinder, redshift)
    print('halofinder', halofinder)

readcat.read_halos(Mmin, pec_vel=pec_vel)
boxsize = readcat.boxsize
px_halo_all = readcat.xh
py_halo_all = readcat.yh
pz_halo_all = readcat.zh

mass_all = readcat.mass
hid_all = readcat.hid

if pec_vel == True:
    vx_halo_all = readcat.vx
    vy_halo_all = readcat.vy
    vz_halo_all = readcat.vz

def calc_one_layer(pz_min, pz_max):
    if sat_from_part == True:
        dsp_part.particle_in_one_layer(pz_min, pz_max)

    sel = (pz_halo_all >= pz_min)&(pz_halo_all < pz_max)
    px_halo_sub = px_halo_all[sel]
    py_halo_sub = py_halo_all[sel]
    pz_halo_sub = pz_halo_all[sel]
    
    if pec_vel == True:
        vx_halo_sub = vx_halo_all[sel]
        vy_halo_sub = vy_halo_all[sel]
        vz_halo_sub = vz_halo_all[sel]

    mass_sub = mass_all[sel]
    hid_sub = hid_all[sel]
    nhalo = len(px_halo_sub)

    #print('nhalo', nhalo)
    hid_out = []
    iscen_out = []
    m_out = []
    px_out = []
    py_out = []
    pz_out = []
    vx_out = []
    vy_out = []
    vz_out = []
    from_part_out = []

    for ih in range(nhalo):
        Ncen, Nsat = Ngal_S20_poisson(mass_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM)
        # first, take care of the central
        if Ncen > 0.5:
            hid_out.append(hid_sub[ih])
            iscen_out.append(1)
            from_part_out.append(1)
            m_out.append(mass_sub[ih])

            px_out.append(px_halo_sub[ih])
            py_out.append(py_halo_sub[ih])
            pz_out.append(pz_halo_sub[ih])
            if pec_vel == True:
                vx_out.append(vx_halo_sub[ih])
                vy_out.append(vy_halo_sub[ih])
                vz_out.append(vz_halo_sub[ih])
            else:
                vx_out.append(0)
                vy_out.append(0)
                vz_out.append(0)

            # then take care of the satellites
            # if mass >= Mmin_part, draw particles; otherwise, draw random numbers
            if Nsat > 0.5:
                
                if sat_from_part == True and mass_sub[ih] >= Mmin_part:
                    px, py, pz, vx, vy, vz = dsp_part.draw_sats(mass_sub[ih], Nsat, px_halo_sub[ih], py_halo_sub[ih], pz_halo_sub[ih])
                    px_out.extend(px)
                    py_out.extend(py)
                    pz_out.extend(pz)
                    vx_out.extend(vx)
                    vy_out.extend(vy)
                    vz_out.extend(vz)
                    from_part_out.extend(np.zeros(Nsat)+1)
                    if len(px) != Nsat: print('problem with part')
                    
                else:
                    px, py, pz = dsp_mc.draw_sat_position(mass_sub[ih], Nsat)
                    vx, vy, vz = dsp_mc.draw_sat_velocity(mass_sub[ih], Nsat)

                    px_out.extend(px_halo_sub[ih] + px)
                    py_out.extend(py_halo_sub[ih] + py)
                    pz_out.extend(pz_halo_sub[ih] + pz)
                    from_part_out.extend(np.zeros(Nsat))

                    if pec_vel == True:
                        vx_out.extend(vx_halo_sub[ih] + vx)
                        vy_out.extend(vy_halo_sub[ih] + vy)
                        vz_out.extend(vz_halo_sub[ih] + vz)
                    else:
                        vx_out.extend(vx)
                        vy_out.extend(vy)
                        vz_out.extend(vz)

                hid_out.extend(np.zeros(Nsat) + hid_sub[ih])
                iscen_out.extend(np.zeros(Nsat))
                m_out.extend(np.zeros(Nsat) + mass_sub[ih])

    px_out = np.array(px_out)
    py_out = np.array(py_out)
    pz_out = np.array(pz_out)
    px_out[px_out < 0] += boxsize
    py_out[py_out < 0] += boxsize
    pz_out[pz_out < 0] += boxsize
    px_out[px_out > boxsize] -= boxsize
    py_out[py_out > boxsize] -= boxsize
    pz_out[pz_out > boxsize] -= boxsize

    vx_out = np.array(vx_out)
    vy_out = np.array(vy_out)
    vz_out = np.array(vz_out)

    print(len(hid_out), len(from_part_out))
    data = np.array([hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, iscen_out, from_part_out]).transpose()
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.dat'
    np.savetxt(ofname, data, fmt='%-12i %-15.12e %-12.12g %-12.12g %-12.12g  %-12.12g %-12.12g %-12.12g %-12i %-12i', header='haloid, M200m, px, py, pz, vx, vy, vz, iscen, from_part') # need a few more decimal places


n_parallel = 100
n_layer = boxsize / n_parallel

def calc_one_bin(ibin):
    pz_min = ibin*n_layer
    pz_max = (ibin+1)*n_layer
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.dat'
    if True:#os.path.exists(ofname) == False:
        calc_one_layer(pz_min=pz_min, pz_max=pz_max)

def merge_files():
    fname_list = glob.glob(f'{out_path}/temp/gals_*.dat')
    hid_out = []
    m_out = []
    px_out = []
    py_out = []
    pz_out = []
    vx_out = []
    vy_out = []
    vz_out = []
    iscen_out = []
    from_part_out = []
    for fname in fname_list:
        data = pd.read_csv(fname, delim_whitespace=True, dtype=np.float64, 
            comment='#', 
            names=['haloid', 'm', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'iscen','from_part'])
        hid_out.extend(data['haloid'])
        m_out.extend(data['m'])
        px_out.extend(data['px'])
        py_out.extend(data['py'])
        pz_out.extend(data['pz'])
        vx_out.extend(data['vx'])
        vy_out.extend(data['vy'])
        vz_out.extend(data['vz'])
        iscen_out.extend(data['iscen'])
        from_part_out.extend(data['from_part'])
    hid_out = np.array(hid_out)
    m_out = np.array(m_out)
    px_out = np.array(px_out)
    py_out = np.array(py_out)
    pz_out = np.array(pz_out)
    vx_out = np.array(vx_out)
    vy_out = np.array(vy_out)
    vz_out = np.array(vz_out)
    iscen_out = np.array(iscen_out)
    from_part_out = np.array(from_part_out)
    sel = np.argsort(-m_out)

    cols=[
      fits.Column(name='haloid', format='K', array=hid_out[sel]),
      fits.Column(name='M200m', format='E', array=m_out[sel]),
      fits.Column(name='px', format='D', array=px_out[sel]),
      fits.Column(name='py', format='D', array=py_out[sel]),
      fits.Column(name='pz', format='D', array=pz_out[sel]),
      fits.Column(name='vx', format='D', array=vx_out[sel]),
      fits.Column(name='vy', format='D', array=vy_out[sel]),
      fits.Column(name='vz', format='D', array=vz_out[sel]),
      fits.Column(name='iscen', format='K',array=iscen_out[sel]),
      fits.Column(name='from_part', format='K',array=from_part_out[sel])
    ]
    coldefs = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    # if sat_from_part == True:
    #     fname = 'gals_from_part.fit'
    # else:
    fname = 'gals.fit'
    tbhdu.writeto(f'{out_path}/{fname}', overwrite=True)

    #os.system(f'rm -rf {out_path}/temp/gals_*.dat')


if __name__ == '__main__':
    #calc_one_bin(0)
    #for i in range(n_parallel):
    #    calc_one_bin(n_parallel)
    
    
    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    start = stop

    with ProcessPoolExecutor() as pool:
        pool.map(calc_one_bin, range(n_parallel))
    stop = timeit.default_timer()
    print('galaxies took', stop - start, 'seconds')
    
    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', stop - start, 'seconds')
    stop = timeit.default_timer()
    dtime = (stop - start_master)/60.
    print(f'total time {dtime:.2g} mins')
    