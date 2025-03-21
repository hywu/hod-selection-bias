#!/usr/bin/env python
import numpy as np
import os, sys
import yaml

# for a given model, calc ngal and Nc
# if Nc too far from DES Y1 value, purge the gals.fit and richness*.fit
# but keep the lensing signal


Nc_target = np.array([762, 376, 123, 91])

output_loc_master = '/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/'
'''
def check_finished_models():
    cosmo_id

    yml_fname = output_loc_master + f'yml/c{cosmo_id:0>3d}_hod{hod_id:0>6d}.yml'

    if os.path.exists(yml_fname):
        with open(yml_fname, 'r') as stream:
            para = yaml.safe_load(stream)

        if para['nbody'] == 'abacus_summit':
            cosmo_id = para.get('cosmo_id', None)
            hod_id = para.get('hod_id', None)
            phase = para.get('phase', None)
            redshift = para['redshift']
            if redshift == 0.3: z_str = '0p300'
            output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
            from hod.utils.get_para_abacus_summit import get_cosmo_para
            cosmo_abacus = get_cosmo_para(cosmo_id)
        else:
           output_loc = para['output_loc']


        model_name = para['model_name']
        rich_name = para['rich_name']
        out_path = f'{output_loc}/model_{model_name}'
        survey = para.get('survey', 'desy1')
        obs_path = f'{out_path}/obs_{rich_name}_{survey}/'

        cmd = f'rm -rf {out_path}/temp/*'
        os.system(cmd)

        fname = obs_path+'abundance.dat'
        os.path.exists(fname)
'''

def check_model(cosmo_id, hod_id):
    yml_fname = output_loc_master + f'yml/c{cosmo_id:0>3d}_hod{hod_id:0>6d}.yml'

    if os.path.exists(yml_fname):
        with open(yml_fname, 'r') as stream:
            para = yaml.safe_load(stream)

        if para['nbody'] == 'abacus_summit':
            cosmo_id = para.get('cosmo_id', None)
            hod_id = para.get('hod_id', None)
            phase = para.get('phase', None)
            redshift = para['redshift']
            if redshift == 0.3: z_str = '0p300'
            output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
            from hod.utils.get_para_abacus_summit import get_cosmo_para
            cosmo_abacus = get_cosmo_para(cosmo_id)
        else:
           output_loc = para['output_loc']


        model_name = para['model_name']
        rich_name = para['rich_name']



        out_path = f'{output_loc}/model_{model_name}'
        survey = para.get('survey', 'desy1')
        obs_path = f'{out_path}/obs_{rich_name}_{survey}/'

        cmd = f'rm -rf {out_path}/temp/*'
        os.system(cmd)

        fname = obs_path+'abundance.dat'
        if os.path.exists(fname):
            x, x, Nc = np.loadtxt(fname, unpack=True)
            Nc_ratio = np.mean(Nc / Nc_target)
            if Nc_ratio < 0.25 or Nc_ratio > 4:
                print(f'unrealistic Nc ratio {Nc_ratio}. purge model: {cosmo_id} {hod_id}')
                cmd1 = f'rm -rf {out_path}/gals.fit'
                cmd2 = f'rm -rf {out_path}/richness_*.fit'
                print(cmd1)
                print(cmd2)
                #os.system(cmd1)
                #os.system(cmd2)
            else:
                print(f'reasonable Nc: {cosmo_id} {hod_id}')

if __name__ == "__main__":
    cosmo_id_list = [0]
    hod_id_list = [0]
    cosmo_id_list.extend(np.zeros(100, dtype=int))
    hod_id_list.extend(np.arange(100, 200, dtype=int))

    for imodel in range(len(cosmo_id_list)):
        cosmo_id = cosmo_id_list[imodel]
        hod_id = hod_id_list[imodel]
        check_model(cosmo_id, hod_id)