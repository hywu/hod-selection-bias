#!/usr/bin/env python
import numpy as np
import os, sys
import subprocess
import yaml

nbody_loc_master = '/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/'
output_loc_master = '/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/'

rich_name = 'd90'
zid = 5
phase = 0

def check_files_needed(zid):
    # check if N-body exists or not
    # if yes, check if the lensing exists or not
    # if no, append to the 'cosmo_id_needed' array
    cosmo_id_needed = []
    cosmo_id_list = [0]
    cosmo_id_list.extend(range(130, 182))
    for cosmo_id in cosmo_id_list:
        nbody_loc = nbody_loc_master + f'base_c{cosmo_id:0>3d}/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/halo_base_c{cosmo_id:0>3d}_ph{phase:0>3d}_z0p{zid}00.h5'
        if os.path.exists(nbody_loc) == False:
            print('missing', nbody_loc)
        else:
            #yml_fname = write_yml(cosmo_id, hod_id)
            model_name = f'hod{hod_id:0>5d}'
            output_loc = output_loc_master + f'base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/'
            out_path = f'{output_loc}/model_{model_name}/'
            lens_fname = f'{out_path}/obs_{rich_name}_desy1/DS_abun_bin_3.dat'
            if os.path.exists(lens_fname) == False:
                cosmo_id_needed.append(cosmo_id)
    return cosmo_id_needed


def write_yml(cosmo_id, hod_id):
    out_string = f"""cosmo_id: {cosmo_id}
hod_id: {hod_id}
model_name: hod{hod_id:0>5d}
rich_name: {rich_name}

nbody: abacus_summit
nbody_loc: {nbody_loc_master}
output_loc: {output_loc_master}
redshift: 0.3
phase: 0

# richness
depth: 90
perc: True
use_rlambda: True
use_pmem: False
pec_vel: True
save_members: False
mdef: vir
sat_from_part: False
"""
    yml_fname = output_loc_master + f'yml/c{cosmo_id:0>3d}_hod{hod_id:0>5d}.yml'
    f = open(yml_fname, "w")
    print(out_string, file=f)
    f.close()

    return yml_fname




if __name__ == "__main__":
    # cosmo_id_list = check_files_needed(zid)
    # print(cosmo_id_list)
    # print('need', len(cosmo_id_list), 'in the job array')

    # i = int(sys.argv[1]) # job array ID
    # print('doing ', cosmo_id)

    job_id = int(sys.argv[1]) # job array ID
    
    # cosmo_list = np.zeros(100, dtype=int)
    # hod_list = np.arange(100, 200, dtype=int)

    # fix cosmo, vary hod
    cosmo_list = np.zeros(100)
    hod_list = 100 + np.arange(100)

    # fix hod, vary cosmo
    # cosmo_list = [0]
    # cosmo_list.extend(range(130, 182))
    # hod_list = np.zeros(53, dtype=int)


    cosmo_hod_list = np.column_stack((cosmo_list, hod_list))

    cosmo_id, hod_id = cosmo_hod_list[job_id]
    cosmo_id = int(cosmo_id)
    hod_id = int(hod_id)
    print('cosmo_id', cosmo_id, 'hod_id', hod_id)

    yml_fname = write_yml(cosmo_id, hod_id)
    with open(yml_fname, 'r') as stream:
        try:
            para = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    #model_name = para['model_name']
    model_name = f'hod{hod_id:0>5d}'

    output_loc = output_loc_master + f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z0p{zid}00/'
    out_path = f'{output_loc}/model_{model_name}/'
    print(out_path)
    if os.path.exists(out_path+'gals.fit'):
        print('galaxies done')
        #print(out_path+'gals.fit')
    else:
        print('need galaxies')
        os.system(f'./make_gal_cat.py {yml_fname}')
        
        
        # subprocess.run(['./make_gal_cat.py', yml_fname], capture_output=True, text=True)
        # print("STDOUT:", result.stdout)  # not saving messages for some reason
        # print("STDERR:", result.stderr)
    
    if os.path.exists(out_path+f'richness_{rich_name}.fit'):
        print('richness done')
        #print(out_path+f'richness_{rich_name}.fit')
    else:
        print('need richness')
        os.system(f'./calc_richness_halo.py {yml_fname}')

        #subprocess.run(['./calc_richness_halo.py', yml_fname], capture_output=True, text=True)
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr), 

    
    survey = para.get('survey', 'desy1')
    obs_path = f'{out_path}/obs_{rich_name}_{survey}/'
    if survey == 'desy1':
        lens_fname = obs_path+'DS_phys_noh_abun_bin_3.dat'
    if survey == 'sdss':
        lens_fname = obs_path+'DS_abun_bin_0.dat'

    if os.path.exists(lens_fname):
        print('lensing done')
    else:
        print('need lensing')
        os.system(f'./plot_lensing.py {yml_fname}')

        #subprocess.run(['./plot_lensing.py', yml_fname], capture_output=True, text=True)
        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr), 
    
    '''
    #### sanity checks ####
    subprocess.run(f'./plot_counts_richness.py {yml_fname}', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    os.chdir('../utils')
    subprocess.run(f'./plot_mor.py {yml_fname}', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    #subprocess.run(f'./plot_hod.py {yml_fname}', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    '''
    