#!/usr/bin/env python
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pandas as pd
import sys
import timeit
import yaml
#import fitsio

start = timeit.default_timer()
start_master = start * 1

#### my functions ####
from hod.utils.read_sim import read_sim
from hod.utils.fid_hod import Ngal_S20_poisson
from hod.utils.get_para_abacus_summit import get_hod_para
from hod.utils.print_memory import print_memory
from hod.utils.merge_files import merge_files
from hod.utils.draw_sat_position import DrawSatPosition

#### read in the yaml file  ####
yml_fname = sys.argv[1]
#./make_gal_cat.py yml/mini_uchuu/mini_uchuu_fid_hod.yml
#./make_gal_cat.py yml/abacus_summit/abacus_summit_template.yml

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


#print_memory(message='after open yml')

redshift = para['redshift']

#### For AbacusSummit ####
if para['nbody'] == 'abacus_summit':
    cosmo_id = para.get('cosmo_id', None)
    hod_id = para.get('hod_id', None)
    phase = para.get('phase', None)
    if redshift == 0.3: z_str = '0p300'
    if redshift == 0.4: z_str = '0p400'
    if redshift == 0.5: z_str = '0p500'
    output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
else:
   output_loc = para['output_loc']

model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}/'

if os.path.isdir(out_path)==False: 
    os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: 
    os.makedirs(out_path+'/temp/')

with open(f'{out_path}/para_gal_cat.yml', 'w') as outfile:
    yaml.dump(para, outfile, default_flow_style=False)

print('output is at ' + out_path)


alpha = para.get('alpha', None) 
if alpha != None: # if parameters are set, use them
    print('read HOD from the yml')
    alpha = para['alpha']
    lgkappa = para['lgkappa']
    lgMcut = para['lgMcut']
    sigmalogM = para['sigmalogM']
    sigmaintr = para['sigmaintr']
    lgM1 = para['lgM1']
    fcen = para.get('fcen', 1.)
else: # if parameters are not set, read from CSV files
    print('read HOD from the csv')
    hod_id = para['hod_id']
    hod_para = get_hod_para(hod_id)
    
    alpha = hod_para['alpha']
    lgkappa = hod_para['lgkappa']
    lgMcut = hod_para['lgMcut']
    sigmalogM = hod_para['sigmalogM']
    sigmaintr = hod_para['sigmaintr']
    lgM1 = hod_para['lgM1']
    fcen = para.get('fcen', 1.)
    
kappa = 10**lgkappa
# lgM20 = para.get('lgM20', None)
# if lgM20 == None:
#     lgM1 = para['lgM1']
# else:
#     M1 = 20**(-1/alpha) *(10**lgM20 - 10**lgkappa * 10**lgMcut)
#     lgM1 = np.log10(M1)

pec_vel = para.get('pec_vel', True)

seed = para.get('seed', 42)
np.random.seed(seed=seed) # for scipy

Mmin = para.get('Mmin', 1e11)

sat_from_part = para.get('sat_from_part', False)
if sat_from_part == True:
    print('drawing sat from particles')
    #from draw_sat_positions_from_particles import DrawSatPositionsFromParticles
    #dsp_part = DrawSatPositionsFromParticles(yml_fname)
    from hod.utils.draw_sat_positions_from_particles_layer import DrawSatPositionsFromParticlesLayer
    dsp_part = DrawSatPositionsFromParticlesLayer(yml_fname)
    Mmin_part = 10**12.5 # only draw particles for halos above this mass

dsp_mc = DrawSatPosition(yml_fname)

readcat = read_sim(para)
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

print_memory(message='done readcat')

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
        Ncen, Nsat = Ngal_S20_poisson(mass_sub[ih], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, sigmalogM=sigmalogM, sigmaintr=sigmaintr, fcen=fcen)
        # first, take care of the central
        if np.isclose(Ncen, 1): # Ncen > 0.5:
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
            if if np.isclose(Nsat, 1) or Nsat > 1: # Nsat > 0.5:
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

    n_workers = int(max(1, n_cpu*0.8))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
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
    
    