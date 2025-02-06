#!/usr/bin/env python
from concurrent.futures import ProcessPoolExecutor
#import h5py
import numpy as np
import os
import pandas as pd
#from astropy.io import fits
import sys
import timeit
import yaml

start = timeit.default_timer()
start_master = start * 1

#### my functions ####
sys.path.append('../utils')
from fid_hod import Ngal_S20_poisson
#from print_memory import print_memory
from merge_files import merge_files

#### read in the yaml file  ####
yml_fname = sys.argv[1]
#./make_gal_cat.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml
#./make_gal_cat.py ../yml/abacus_summit/abacus_summit_fid_hod.yml

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#print_memory(message='after open yml')


#### For AbacusSummit ####
if para['nbody'] == 'abacus_summit':
    cosmo_id = para.get('cosmo_id', None)
    hod_id = para.get('hod_id', None)
    phase = para.get('phase', None)
    redshift = para['redshift']
    if redshift == 0.3: z_str = '0p300'
    output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
else:
   output_loc = para['output_loc']


model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}/'

def get_hod_para(hod_id_wanted):
    df = pd.read_csv('/projects/hywu/cluster_sims/cluster_finding/work/hod/repo/abacus_summit/hod_params.csv', sep=',')
    nrows = df.shape[0]
    for irow in np.arange(nrows):
        row = df.iloc[irow]
        hod_id = row['hod_id']
        if hod_id == hod_id_wanted:
            row_output = row
            break
    return row_output # it's a data frame, but it can be used as a dictionary


alpha = para.get('alpha', None)
if alpha != None:
    alpha = para['alpha']
    lgkappa = para['lgkappa']
    lgMcut = para['lgMcut']
    sigmalogM = para['sigmalogM']
    sigma_intr = para['sigmaintr']
    lgM1 = para['lgM1']
else:
    hod_id = para['hod_id']
    hod_para = get_hod_para(hod_id)
    alpha = hod_para['alpha']
    lgkappa = hod_para['lgkappa']
    lgMcut = hod_para['lgMcut']
    sigmalogM = hod_para['sigmalogM']
    sigma_intr = hod_para['sigmaintr']
    lgM1 = hod_para['lgM1']

kappa = 10**lgkappa
# lgM20 = para.get('lgM20', None)
# if lgM20 == None:
#     lgM1 = para['lgM1']
# else:
#     M1 = 20**(-1/alpha) *(10**lgM20 - 10**lgkappa * 10**lgMcut)
#     lgM1 = np.log10(M1)

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

#print_memory(message='before readcat')

if para['nbody'] == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    readcat = ReadMiniUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'uchuu':
    from read_uchuu import ReadUchuu
    readcat = ReadUchuu(para['nbody_loc'], redshift)

if para['nbody'] == 'abacus_summit':
    from read_abacus_summit import ReadAbacusSummit
    readcat = ReadAbacusSummit(para['nbody_loc'], redshift, cosmo_id=cosmo_id)

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

#print_memory(message='done readcat')

def calc_one_layer(pz_min, pz_max):
    #print_memory(message='before calc_one_layer')
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
            m_out.append(mass_sub[ih])

            if sat_from_part == True:
                from_part_out.append(1)
            else:
                from_part_out.append(0)

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

    #print(len(hid_out), len(from_part_out))
    data = np.array([hid_out, m_out, px_out, py_out, pz_out, vx_out, vy_out, vz_out, iscen_out, from_part_out]).transpose()
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.dat'
    np.savetxt(ofname, data, fmt='%-12i %-15.12e %-12.12g %-12.12g %-12.12g  %-12.12g %-12.12g %-12.12g %-12i %-12i', header='haloid mass px py pz vx vy vz iscen from_part', comments='') # need a few more decimal places


    #print_memory(message='done calc_one_layer')

n_parallel = 100
n_layer = boxsize / n_parallel

def calc_one_bin(ibin):
    pz_min = ibin*n_layer
    pz_max = (ibin+1)*n_layer
    ofname = f'{out_path}/temp/gals_{pz_min}_{pz_max}.dat'
    if True:#os.path.exists(ofname) == False:
        calc_one_layer(pz_min=pz_min, pz_max=pz_max)





# def merge_files_old():
#     fname_list = glob.glob(f'{out_path}/temp/gals_*.dat')
#     hid_out = []
#     m_out = []
#     px_out = []
#     py_out = []
#     pz_out = []
#     vx_out = []
#     vy_out = []
#     vz_out = []
#     iscen_out = []
#     from_part_out = []
#     for fname in fname_list:
#         data = pd.read_csv(fname, sep=r'\s+', dtype=np.float64, 
#             comment='#', 
#             names=['haloid', 'm', 'px', 'py', 'pz', 'vx', 'vy', 'vz', 'iscen','from_part'])
#         hid_out.extend(data['haloid'])
#         m_out.extend(data['m'])
#         px_out.extend(data['px'])
#         py_out.extend(data['py'])
#         pz_out.extend(data['pz'])
#         vx_out.extend(data['vx'])
#         vy_out.extend(data['vy'])
#         vz_out.extend(data['vz'])
#         iscen_out.extend(data['iscen'])
#         from_part_out.extend(data['from_part'])
#     hid_out = np.array(hid_out)
#     m_out = np.array(m_out)
#     px_out = np.array(px_out)
#     py_out = np.array(py_out)
#     pz_out = np.array(pz_out)
#     vx_out = np.array(vx_out)
#     vy_out = np.array(vy_out)
#     vz_out = np.array(vz_out)
#     iscen_out = np.array(iscen_out)
#     from_part_out = np.array(from_part_out)
#     sel = np.argsort(-m_out)

#     cols=[
#       fits.Column(name='haloid', format='K', array=hid_out[sel]),
#       fits.Column(name='M200m', format='E', array=m_out[sel]),
#       fits.Column(name='px', format='D', array=px_out[sel]),
#       fits.Column(name='py', format='D', array=py_out[sel]),
#       fits.Column(name='pz', format='D', array=pz_out[sel]),
#       fits.Column(name='vx', format='D', array=vx_out[sel]),
#       fits.Column(name='vy', format='D', array=vy_out[sel]),
#       fits.Column(name='vz', format='D', array=vz_out[sel]),
#       fits.Column(name='iscen', format='K',array=iscen_out[sel]),
#       fits.Column(name='from_part', format='K',array=from_part_out[sel])
#     ]
#     coldefs = fits.ColDefs(cols)
#     tbhdu = fits.BinTableHDU.from_columns(coldefs)
#     fname = 'gals.fit'
#     tbhdu.writeto(f'{out_path}/{fname}', overwrite=True)

#     #os.system(f'rm -rf {out_path}/temp/gals_*.dat')


if __name__ == '__main__':
    #calc_one_bin(0)

    stop = timeit.default_timer()
    print('make_gal_cat.py prep took', '%.2g'%((stop - start)/60), 'mins')
    
    start = stop
    n_cpu = os.getenv('SLURM_CPUS_PER_TASK') # os.cpu_count() not working
    if n_cpu is not None:
        n_cpu = int(n_cpu)
        print(f'Assigned CPUs: {n_cpu}') 
    else:
        print('Not running under SLURM or the variable is not set.') 
        n_cpu = 1

    with ProcessPoolExecutor(max_workers=n_cpu) as pool:
        for result in pool.map(calc_one_bin, range(n_parallel)):
            if result: print(result)  # output error
    stop = timeit.default_timer()
    print('galaxies took', '%.2g'%((stop - start)/60), 'mins')


    start = stop
    merge_files(in_fname=f'{out_path}/temp/gals_*.dat', out_fname=f'{out_path}/gals.fit', nfiles_expected=n_parallel)
    stop = timeit.default_timer()
    print('merging took', '%.2g'%((stop - start)/60), 'mins')

    stop = timeit.default_timer()
    dtime = (stop - start_master)/60.
    print(f'total time {dtime:.2g} mins')
    