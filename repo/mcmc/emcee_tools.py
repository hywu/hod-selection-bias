#!/usr/bin/env python
# This file defines general log-likelihood class for MCMC fitting
# and functions to setup and call emcee to conduct the fitting.
import numpy as np
import emcee
import os

class LnLikelihood(object):
    '''This class defines the logarithmic likelihood used for MCMC;
    fitting.  Inputs: data_vec, cov: arrays; data vector and cov matrix;
    model: function, the model to be fit;
    params_free_range: range of flat prior
    params_free_name: dictionary of free parameters;
    paramsfixed: values of fixed parameters;
    paramsdict: dictionary of fixed parameters

    '''

    def __init__(self, data_vec, cov, model, params_free_range,
                 params_free_name, params_fixed_value=None, params_fixed_name=None):

        #def __init__(self, data_x, data_y, cov, model, params_free_range,
        #         params_free_name, params_fixed_value=None, params_fixed_name=None):

        #self.data_x = data_x
        #self.data_y = data_y
        self.data_vec = data_vec
        self.model = model
        self.cov = cov
        self.params_free_range = params_free_range
        self.params_free_name = params_free_name
        self.nparams = len(params_free_name)

        self.params_fixed_value = params_fixed_value
        self.params_fixed_name = params_fixed_name

        self.invcov = np.linalg.inv(cov)

    def lnprior(self, params):
        for i, p in enumerate(params):
            if not(self.params_free_range[i][0] <= p <= self.params_free_range[i][1]):
                return -np.inf
            else:
                continue
        return 0

    def lnposterior(self, params, **kwargs):

        prior = self.lnprior(params)
        if prior == -np.inf:
            return prior


        #model_vec = self.model(self.data_x, params, self.params_free_name,
        #                     self.params_fixed_value, self.params_fixed_name)
        #model_vec = self.model(self.data_x, params)
        model_vec = self.model(params)
        

        if (model_vec is None) or (self.cov is None):
            return -np.inf
        elif  (np.isnan(model_vec).sum() != 0) or (np.isnan(self.cov).sum() != 0):
            return -np.inf
        else:

            diff = model_vec - self.data_vec
            chi2 = np.dot(np.dot(diff.T, self.invcov), diff)*0.5
            if np.isnan(chi2):
                return -np.inf                
            if chi2 < 0:
                return -np.inf
            #print(chi2 / (self.data_x.size-len(params)))
            return -chi2# - 0.5*np.linalg.slogdet(cov)[1]


#def mcmc_setup(filename):



def runmcmc(params0, nstep, nwalkers, lstep, lnposterior, out_file, pool=None, burnin=None, thread=1, **kwargs):

    ndim = params0.size
    
    # copied from 
    # https://github.com/frankelzeng/selection_bias/blob/main/source/emcee_wpmm_30bins_cov_ab.py
    if os.path.exists(out_file):
        print('continuing the chain')
        backend = emcee.backends.HDFBackend(out_file)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, backend=backend)
        sampler.run_mcmc(None, nstep)
    else:
        print('new chain')
        backend = emcee.backends.HDFBackend(out_file)
        backend.reset(nwalkers, ndim)
        params0[np.abs(params0)<1e-3] = 1e-3 # avoid zoros
        pos = [np.array(params0)*(1 + lstep * np.random.rand(ndim)) for i in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, backend=backend)
        pos, prob, state = sampler.run_mcmc(pos, 1) 
        sampler.reset() # what's the purpose of this?
        sampler.run_mcmc(pos, nstep, rstate0 = state)

    ## TODO: deal with pool and thread

    '''
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, threads=60)
    if pool is None:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, backend=backend, threads=thread)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, backend=backend, pool=pool)
    #pos = [np.asarray(params0) + lstep * np.random.rand(ndim) for i in range(nwalkers)]

    if burnin is None:
        pos, prob, state = sampler.run_mcmc(pos, nstep)#, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
    else:
        pos, prob, stata = sampler.run_mcmc(pos, burnin)#, progress=True)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)#, progress=True)
        return sampler.flatchain, sampler.flatlnprobability
    '''
    return sampler.flatchain, sampler.flatlnprobability