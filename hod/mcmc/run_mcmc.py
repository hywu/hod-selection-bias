#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import os, sys
import emcee
from get_model import GetModel
from get_data_vector import get_data_vector


'''
AttributeError: 'EnsembleSampler' object has no attribute '_previous_state'
==> delete the old h5 file!
'''

# ./run_mcmc.py s8Omhod all abacus_summit q180_bg_miscen abun lensing 5k10 0 0 

para_name = sys.argv[1] # 's8Omhod'
emu_name = sys.argv[2]  # 'all', 'narrow'
observation = sys.argv[3] # 'abacus_summit', 'flamingo'
rich_name = sys.argv[4] # 'q180_bg_miscen', 'redmapper'
binning = sys.argv[5] # 'lam', 'abun'
data_vector_name = sys.argv[6] # 'counts', 'lensing', 'counts_lensing'
cov_name = sys.argv[7] # '5k10', '5k30', '15k10', '15k30'
iz = int(sys.argv[8]) # 0
run_id = int(sys.argv[9]) # 0

if data_vector_name == 'counts':
    data_vector = ['counts']
if data_vector_name == 'lensing':
    data_vector = ['lensing']
if data_vector_name == 'counts_lensing':
    data_vector = ['counts', 'lensing']

z_list = [0.3, 0.4, 0.5]
redshift = z_list[iz]

# Parse the yaml file
yml_name = f'yml/emcee_{para_name}_{observation}.yml'
from parse_yml import ParseYml
parse = ParseYml(yml_name)
nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()

out_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_mcmc/{observation}_{rich_name}_{binning}_{data_vector_name}/'
plot_loc = f'../../plots/mcmc/{observation}_{rich_name}_{binning}_{data_vector_name}/'
chain_name = f'{para_name}_{emu_name}_{cov_name}_z{redshift}_run{run_id}'
out_file = f'{out_loc}/mcmc_{chain_name}.h5'
print('output: ', out_file)

if os.path.isdir(out_loc) == False:
    os.makedirs(out_loc)
if os.path.isdir(plot_loc) == False:
    os.makedirs(plot_loc)

# save a back-up parameter file
import yaml
with open(yml_name, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(f'{out_loc}/para_{chain_name}.yml', 'w') as outfile:
    yaml.dump(para, outfile)


if __name__ == "__main__":

    if cov_name in ['5k10', '5k30']:
        survey_area = 5_000.
    elif cov_name in ['15k10', '15k30']:
        survey_area = 15_000.
    else:
        survey_area = 1437.

    # load the dta vector
    data_vec, cov = get_data_vector(observation, rich_name, binning, iz, data_vector=data_vector, survey_area=survey_area, cov_name=cov_name)
    print(data_vector)
    print('data vector size', len(data_vec))

    # define the likelihood
    from emcee_tools import LnLikelihood, runmcmc
    gm = GetModel(emu_name, binning, iz, observation, params_free_name, params_fixed_value, params_fixed_name, data_vector=data_vector, survey_area=survey_area)
    lnlike = LnLikelihood(data_vec, cov, gm.model, params_range,
                                 params_free_name, params_fixed_value, params_fixed_name)

    # run MCMC
    pool = None
    mcmc_chain, posterior = runmcmc(params_free_ini, nsteps, nwalkers, lsteps, 
                                       lnlike.lnposterior, out_file,
                                       pool, burnin=burnin)

    # plot MCMC posterior! 
    os.system(f'./plot_mcmc.py {para_name} {emu_name} {observation} {rich_name} {binning} {data_vector_name} {cov_name} {iz} {run_id} 2>&1 | tee sbatch_output/{emu_name}.out')
