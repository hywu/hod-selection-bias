#!/usr/bin/env python
import os, sys
import numpy as np

import emcee
from getdist import MCSamples, plots

#from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig, ChainConfig
sys.path.append('../mcmc')
from hod.mcmc.parse_yml import ParseYml

def compare_mcmc_getdist(yml_name, outfile_list, label_list, color_list, filled_list=True, cosmo_only=False, params_limit_dict=False):
    
    parse = ParseYml(yml_name)
    nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()

    params_free_label = parse.params_free_label
    if cosmo_only == True:
        params_free_name = params_free_name[0:2]
        params_free_label = params_free_label[0:2]
    
    #c = ChainConsumer()
    sample_list = []
    line_args_list = []
    for i, out_file in enumerate(outfile_list):
        if os.path.exists(out_file) == False:
            print('missing' + out_file)
        else:
            #print(out_file)
            label = label_list[i]
            color = color_list[i]
            reader = emcee.backends.HDFBackend(out_file, read_only=True)
            chain = reader.get_chain(discard=1000, thin=10) # Shape: (nsteps, nwalkers, ndim)
            
            if cosmo_only == True:
                chain = chain[:, :, 0:2]

            logprob = reader.get_log_prob(discard=1000, thin=10) # Shape: (nsteps, nwalkers) # thin: take every 10 steps
            chain_list = [chain[:, i, :] for i in range(chain.shape[1])]  # List of (nsteps, ndim)
            logprob_list = [logprob[:, i] for i in range(logprob.shape[1])]  # List of (nsteps,)

            print('np.shape(chain_list)')
            print(np.shape(chain_list))
            print(np.shape(logprob_list))


            params_free_label = [s.replace("$", "") for s in params_free_label]
            samples = MCSamples(
                samples=chain_list,  # List of arrays, each walker as separate chain
                loglikes=[-lp for lp in logprob_list],  # List of -log(posterior) arrays
                names=params_free_name,
                labels=params_free_label,
            )
            sample_list.append(samples)
            line_args_list.append({'color': color})

    # dictionary for the truth value (called 'markers')
    truth_loc = dict(zip(parse.params_free_name, parse.params_free_truth.astype(float)))
    #print(parse.params_free_truth)

    # dictionary for the axes limit # example: param_limits={"sigma8": (0, 1), "alpha": (0, 1)}
    if params_limit_dict == False:
        params_limit_dict = dict(zip(parse.params_free_name[0:2], params_range[0:2])) # only sigma8, Om
        #print(params_limit_dict)

    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    #g.settings.alpha_filled_add = 0.4
    g.settings.title_limit_fontsize = 18
    g.settings.axes_fontsize = 20
    g.settings.axes_labelsize = 30
    g.settings.legend_fontsize = 30
    g.triangle_plot(sample_list, 
        legend_labels=label_list,
        legend_loc="upper right",
        contour_colors=color_list,
        contour_lws = 2,
        markers=truth_loc,
        marker_args={"lw": 1, 'color':'k'},#}, , 'ls': '-'
        #line_args=[{"ls": "--", "color": "green"}, {"lw": 2, "color": "darkblue"}]
        #line_args={"lw": 1.5},
        line_args=line_args_list,
        filled=filled_list,
        param_limits=params_limit_dict
        )
        #title_limit=1,  # first title limit (for 1D plots) is 68% by default
        