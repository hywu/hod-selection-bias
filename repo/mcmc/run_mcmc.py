#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import emcee
import os, sys

run_name = sys.argv[1] #'s8Omns'#'s8Om'

yml_name = f'yml/emcee_{run_name}.yml'

#### Parse the Yaml file
from emcee_tools import ParseYaml
parse = ParseYaml(yml_name)
nsteps, nwalkers, lsteps, burnin, paramsdict_free, params_free_0, params_range,\
        paramsdict_fixed, params_fixed = parse.parse_yaml()

#### choose which emulator to use based on the parameters
if 'sigma8' in paramsdict_free and 'alpha' in paramsdict_fixed:
    emu_name = 'fixhod'
elif 'alpha' in paramsdict_free and 'sigma8' in paramsdict_fixed:
    emu_name = 'fixcos'
else: #  'alpha' in paramsdict_free and 'sigma8' in paramsdict_free:
    emu_name = 'all'

print('emu_name', emu_name)


out_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_train/{emu_name}/mcmc/'
plot_loc = f'../../plots/emulator/{emu_name}/'
if os.path.isdir(out_loc) == False:
    os.makedirs(out_loc)

out_file = f'{out_loc}/mcmc_{run_name}.h5'


print('output: ', out_file)


#### Define the model class
sys.path.append('../emulator')
from s3_pred_radius import PredDataVector
pdv = PredDataVector(emu_name)

class GetModel(object): 
    def __init__(self, paramsdict_free, params_fixed,
                 paramsdict_fixed, **kwargs):
        self.paramsdict_fixed = paramsdict_fixed
        self.paramsdict_free = paramsdict_free
        self.params_fixed = params_fixed
       
    def get_kw(self, params):
        kw = {} # if there's extra keyword
        for i, pf in enumerate(self.paramsdict_fixed):
            kw[pf] = self.params_fixed[i]

        for i, pf in enumerate(self.paramsdict_free):
            kw[pf] = params[i]
        ###
        self.kw = kw
    
    def model(self, params):
        self.get_kw(params)
        params = self.kw
        #print('params inside model', params)
        sigma8 = params['sigma8']
        OmegaM = params['OmegaM']
        ns = params['ns']
        OmegaB = params['OmegaB']
        w0 = params['w0']
        wa = params['wa']
        #sigma8, OmegaM, ns, Ob0, w0, wa, Nur, alpha_s
        cosmo_para = np.array([sigma8, OmegaM, ns, OmegaB, w0, wa, 2.0328, 0])
        
        alpha = params['alpha']
        lgM1 = params['lgM1']
        lgMcut = params['lgMcut']
        # alpha,lgM1,lgkappa,lgMcut,sigmalogM
        hod_para = np.array([alpha, lgM1, 0, lgMcut, 0.1]) #1,12.9,0,11.7,0.1


        X_input = np.append(cosmo_para, hod_para)
        model = np.append(pdv.pred_abundance(X_input), pdv.pred_lensing(X_input))
        return model


#### Get the data
data_vec = np.loadtxt('../emulator/data/data_vector.dat')
cov_inv = np.loadtxt('../emulator/data/cov_inv.dat')
data_cov = np.linalg.inv(cov_inv)

#### Define the likelihood
from emcee_tools import LnLikelihood, runmcmc
gm = GetModel(paramsdict_free, params_fixed, paramsdict_fixed)
lnlike = LnLikelihood(data_vec, data_cov, gm.model, params_range,
                             paramsdict_free, params_fixed, paramsdict_fixed)

#### Run MCMC
pool=None

mcmc_chain, posterior = runmcmc(params_free_0, nsteps, nwalkers, lsteps, 
                                   lnlike.lnposterior, out_file,
                                   pool, burnin=burnin)

#### Plot posterior
ndim = params_free_0.size
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
reader = emcee.backends.HDFBackend(out_file, read_only=True)
samples = reader.get_chain()
print('np.shape(samples) =', np.shape(samples))

labels = parse.params_label_free
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
    
axes[-1].set_xlabel("step number");

plt.savefig(plot_loc+f'samples_{run_name}.pdf')


import corner
flat_samples = reader.get_chain(discard=100, thin=10, flat=True)
print('after thinning', flat_samples.shape)
truth = params_free_0
fig = corner.corner(flat_samples, labels=labels)
# add the truth (there must be a better way)
axes = np.array(fig.axes).reshape((ndim, ndim))
for i in range(ndim):
    ax = axes[i, i]
    ax.axvline(truth[i])

plt.savefig(plot_loc+f'mcmc_{run_name}.pdf')



