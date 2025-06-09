#!/usr/bin/env python
import yaml
import numpy as np
import os



def get_flamingo_cosmo(sim_name):
    input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/' # cosma
    if os.path.exists(input_loc) == False:
        input_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/flamingo/output_{sim_name}/' # m3

    yml_fname = input_loc + 'used_parameters.yml'
    
    with open(yml_fname, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    cosmo = parsed_yaml['Cosmology']
    return cosmo


def get_snap_name(sim_name, redshift):
    input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/' # cosma
    if os.path.exists(input_loc) == False:
        input_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/flamingo/output_{sim_name}/' # m3

    #### identify the snapshot ID
    z_output = np.loadtxt(input_loc + 'output_list.txt')
    snap_id_list = np.arange(len(z_output))
    idx = np.argmin(abs(z_output-redshift))
    snap_id = snap_id_list[idx]
    snap_name = f'{snap_id:0>4d}'
    #print('z=0.3 snap id', snap_name)
    return snap_name


if __name__ == "__main__":
    print(get_flamingo_cosmo(sim_name='L1000N3600/HYDRO_FIDUCIAL/'))
    print(get_snap_name(sim_name='L1000N3600/HYDRO_FIDUCIAL/', redshift=0.3))
