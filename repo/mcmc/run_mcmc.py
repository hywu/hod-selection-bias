#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import emcee
import os, sys

para_name = 's8Omhod' #'template'#
data_name = 'flamingo' # 'abacus_summit' #
run_name = para_name

#run_name = sys.argv[1] #'s8Omns'#'s8Om'

yml_name = f'yml/emcee_{run_name}_{data_name}.yml'

#### Parse the Yaml file
from parse_yml import ParseYml
parse = ParseYml(yml_name)
nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()

#### choose which emulator to use based on the parameters
if 'sigma8' in params_free_name and 'alpha' in params_fixed_name:
    emu_name = 'fixhod'
elif 'alpha' in params_free_name and 'sigma8' in params_fixed_name:
    emu_name = 'fixcos'
else: #  'alpha' in params_free_name and 'sigma8' in params_free_name:
    emu_name = 'all'

print('emu_name', emu_name)


out_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_train/{emu_name}/mcmc_{data_name}/'
plot_loc = f'../../plots/emulator/{emu_name}/{data_name}/'
if os.path.isdir(out_loc) == False:
    os.makedirs(out_loc)
if os.path.isdir(plot_loc) == False:
    os.makedirs(plot_loc)

out_file = f'{out_loc}/mcmc_{run_name}.h5'


print('output: ', out_file)


#### Define the model class
sys.path.append('../emulator')
from s3_pred_radius import PredDataVector
pdv = PredDataVector(emu_name)

class GetModel(object): 
    def __init__(self, params_free_name, params_fixed_value,
                 params_fixed_name, **kwargs):
        self.params_fixed_name = params_fixed_name
        self.params_free_name = params_free_name
        self.params_fixed_value = params_fixed_value
       
    def get_kw(self, params):
        kw = {} # if there's extra keyword
        for i, pf in enumerate(self.params_fixed_name):
            kw[pf] = self.params_fixed_value[i]

        for i, pf in enumerate(self.params_free_name):
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


if __name__ == "__main__":
    #### Get the data
    data_vec = np.loadtxt(f'../emulator/data_vector_{data_name}/data_vector.dat')
    cov_inv = np.loadtxt(f'../emulator/data_vector_{data_name}/cov_inv.dat')
    data_cov = np.linalg.inv(cov_inv)

    #### Define the likelihood
    from emcee_tools import LnLikelihood, runmcmc
    gm = GetModel(params_free_name, params_fixed_value, params_fixed_name)
    lnlike = LnLikelihood(data_vec, data_cov, gm.model, params_range,
                                 params_free_name, params_fixed_value, params_fixed_name)

    #### Run MCMC
    pool=None

    mcmc_chain, posterior = runmcmc(params_free_ini, nsteps, nwalkers, lsteps, 
                                       lnlike.lnposterior, out_file,
                                       pool, burnin=burnin)

    #### Plot posterior distribution
    ndim = params_free_ini.size
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    reader = emcee.backends.HDFBackend(out_file, read_only=True)
    samples = reader.get_chain()
    print('np.shape(samples) =', np.shape(samples))

    labels = parse.params_free_label
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        
    axes[-1].set_xlabel("step number");

    plt.savefig(plot_loc+f'samples_{run_name}.pdf', dpi=72)


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

    plt.savefig(plot_loc+f'mcmc_{run_name}.pdf', dpi=72)
    print('plots saved at ' + plot_loc)


