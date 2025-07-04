#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import os, sys
import emcee
from get_model import GetModel
from get_data_vector import get_data_vector

# ./run_mcmc.py s8Omhod all abacus_summit q180_bg_miscen abun lensing desy1thre 0 0 
para_name = sys.argv[1] #'s8Omhod'
emu_name = sys.argv[2]  #'all' # 'narrow'
data_name = sys.argv[3] #'abacus_summit' #'flamingo'
rich_name = sys.argv[4] #'q180_bg_miscen'
binning = sys.argv[5] # 'lam', 'abun'
data_vector_name = sys.argv[6] # 'counts', 'lensing'
cov_name = sys.argv[7] # 'desy1', 'area10k', 'nsrc50', 'desy1thre'
iz = int(sys.argv[8])
run_id = int(sys.argv[9])
survey = cov_name

if data_vector_name == 'counts':
    data_vector = ['counts']
if data_vector_name == 'lensing':
    data_vector = ['lensing']
if data_vector_name == 'counts_lensing':
    data_vector = ['counts', 'lensing']

z_list = [0.3, 0.4, 0.5]
redshift = z_list[iz]

# Parse the Yaml file
yml_name = f'yml/emcee_{para_name}_{data_name}.yml'
from parse_yml import ParseYml
parse = ParseYml(yml_name)
nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()


out_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_mcmc/{emu_name}/mcmc_{data_name}/{survey}_{binning}/{data_vector_name}/'
plot_loc = f'../../plots/mcmc/{emu_name}/{data_name}/{survey}_{binning}/{data_vector_name}/'
if os.path.isdir(out_loc) == False:
    os.makedirs(out_loc)
if os.path.isdir(plot_loc) == False:
    os.makedirs(plot_loc)

out_file = f'{out_loc}/mcmc_{para_name}_{rich_name}_z{redshift}_run{run_id}.h5'
print('output: ', out_file)

# save a back-up parameter file
import yaml
with open(yml_name, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(f'{out_loc}/para_{para_name}_{rich_name}_z{redshift}_run{run_id}.yml', 'w') as outfile:
    yaml.dump(para, outfile)





if __name__ == "__main__":

    if cov_name=='area10k':
        survey_area = 10_000.
    else:
        survey_area = 1437.

    data_vec, cov = get_data_vector(data_name, rich_name, binning, iz, survey=survey, data_vector=data_vector, survey_area=survey_area, cov_name=cov_name)
    print(data_vector)
    print('data vector size', len(data_vec))

    # data_vec = np.loadtxt(f'../data_vector/data_vector_{data_name}/data_vector_{rich_name}_{binning}_z{redshift}.dat')
    # cov = np.loadtxt(f'../data_vector/data_vector_abacus_summit/cov_z{redshift}.dat')
    # if binning == 'abun':
    #     data_vec = data_vec[4:]
    #     cov = cov[4:, 4:]

    #### Define the likelihood
    from emcee_tools import LnLikelihood, runmcmc
    gm = GetModel(emu_name, binning, iz, survey, params_free_name, params_fixed_value, params_fixed_name, data_vector=data_vector, survey_area=survey_area)
    lnlike = LnLikelihood(data_vec, cov, gm.model, params_range,
                                 params_free_name, params_fixed_value, params_fixed_name)

    #### Run MCMC
    pool = None

    mcmc_chain, posterior = runmcmc(params_free_ini, nsteps, nwalkers, lsteps, 
                                       lnlike.lnposterior, out_file,
                                       pool, burnin=burnin)

    #### Plot MCMC! 
    os.system(f'./plot_mcmc.py {para_name} {emu_name} {data_name} {rich_name} {binning} {data_vector_name} {cov_name} {iz} {run_id} 2>&1 | tee sbatch_output/{emu_name}.out')

    '''
    # moved to plot_mcmc.py
    #### Plot posterior distribution
    ndim = params_free_ini.size
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    reader = emcee.backends.HDFBackend(out_file, read_only=True)
    samples = reader.get_chain()
    print('np.shape(samples) =', np.shape(samples))
    labels = parse.params_free_label
    '''
    '''
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
    axes[-1].set_xlabel("step number");

    plt.savefig(plot_loc+f'samples_{emu_name}_z0p{zid}00.pdf', dpi=72)
    '''
    '''
    import corner
    flat_samples = reader.get_chain(discard=100, thin=10, flat=True)
    print('after thinning', flat_samples.shape)
    truth = params_free_ini
    fig = corner.corner(flat_samples, labels=labels)

    if data_name == 'abacus_summit':
        # add the truth (there must be a better way)
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(truth[i])

    plt.savefig(plot_loc+f'mcmc_{para_name}_z{redshift}_run{run_id}.pdf', dpi=72)
    print('plots saved at ' + plot_loc)
    '''

