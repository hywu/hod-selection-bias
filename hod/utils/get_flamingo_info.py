#!/usr/bin/env python
import yaml
import numpy as np

def get_flamingo_cosmo(sim_name):
    loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
    yml_fname = loc + 'used_parameters.yml'
    
    with open(yml_fname, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    cosmo = parsed_yaml['Cosmology']
    return cosmo


def get_snap_name(sim_name, redshift):
    #### identify the snapshot ID
    input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
    z_output = np.loadtxt(input_loc + 'output_list.txt')
    snap_id_list = np.arange(len(z_output))
    idx = np.argmin(abs(z_output-redshift))
    snap_id = snap_id_list[idx]
    snap_name = f'{snap_id:0>4d}'
    #print('z=0.3 snap id', snap_name)
    return snap_name