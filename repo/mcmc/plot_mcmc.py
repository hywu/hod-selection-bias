#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import pandas as pd
import os, sys
import emcee
from chainconsumer import Chain, ChainConsumer, ChainConfig
from get_model import GetModel

# ./plot_mcmc.py s8Omhod all abacus_summit q180_bg_miscen abun 0 0 
para_name = sys.argv[1] #'s8Omhod'
emu_name = sys.argv[2] #'wide' # 'narrow'
data_name = sys.argv[3] #'abacus_summit' #'flamingo'
rich_name = sys.argv[4] #'q180_bg_miscen'
binning = sys.argv[5]
iz = int(sys.argv[6])
run_id = int(sys.argv[7])


z_list = [0.3, 0.4, 0.5]
redshift = z_list[iz]

# Parse the Yaml file
yml_name = f'yml/emcee_{para_name}_{data_name}.yml'
from parse_yml import ParseYml
parse = ParseYml(yml_name)
nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()



out_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_mcmc/{emu_name}/mcmc_{data_name}/{binning}/'
plot_loc = f'../../plots/mcmc/{emu_name}/{data_name}/{binning}/'

out_file = f'{out_loc}/mcmc_{para_name}_{rich_name}_z{redshift}_run{run_id}.h5'
print('output: ', out_file)

reader = emcee.backends.HDFBackend(out_file, read_only=True)



from chainconsumer import Chain, ChainConsumer, Truth, PlotConfig
chain = Chain.from_emcee(reader, columns=parse.params_free_name, 
                         name=f'{emu_name} {data_name}', color="blue", show_label_in_legend=True)
c = ChainConsumer()
c.add_chain(chain)
# example
# https://samreay.github.io/ChainConsumer/generated/gallery/advanced_examples/plot_5_multiple_truth_values/


#### plot the truth # set line wideth manually in the end
loc = dict(zip(parse.params_free_name, parse.params_free_truth))
c.add_truth(Truth(location=loc, color='k'))#, linestyle="-", linewidth=20))

#### plot the prior ranges
para_min = dict(zip(parse.params_free_name, parse.params_free_range[:,0]))
para_max = dict(zip(parse.params_free_name, parse.params_free_range[:,1]))
c.add_truth(Truth(location=para_min, color='C0'))#, linestyle=":", linewidth=0.5))
c.add_truth(Truth(location=para_max, color='C0'))#, linestyle=":", linewidth=0.5))

#### latex label, fontsize
lab = dict(zip(parse.params_free_name, parse.params_free_label))
pc = PlotConfig(
        labels=lab,
        label_font_size = 30,
        summary_font_size = 20,
        tick_font_size = 15,
        #tick_font = 'roman'
)
c.set_plot_config(pc)
c.set_override(ChainConfig(shade_alpha=0.5))
#c.set_override(ChainConfig(show_contour_labels=True)) % 68% and 95%
fig = c.plotter.plot()

#### Plot the emulator range
loc = '/projects/hywu/cluster_sims/cluster_finding/data/'
train_loc = loc + f'emulator_train/{emu_name}/z0p{3+iz}00/{binning}/'
df = pd.read_csv(f'{train_loc}/parameters.csv')
df = df[parse.params_free_name]


ndim = len(parse.params_free_name)
data = df.to_numpy()
for i in range(0, ndim):
    for j in range(0, ndim):
        ax = plt.subplot(ndim, ndim,1+ndim*j+i)
        if j > i:
            ax.scatter(data[:, i], data[:, j], s=5, alpha=1, color='k')
        if True:
            # also set up plotting style
            ax.grid(False)
            for line in ax.lines:
                #if line.get_linestyle() == "--":
                line.set_linewidth(2)
                
plt.suptitle(f'{data_name} {rich_name} {binning} z={redshift}')

plt.savefig(plot_loc+f'mcmc_{para_name}_{rich_name}_z{redshift}_run{run_id}.pdf', dpi=72)

###################################################################
#### Plot posterior prediction
# Sample 5000 points from the chain
# Use emulator to calculate the posterior
flat_samples = reader.get_chain(discard=100, thin=10, flat=True)
print('after thinning', flat_samples.shape)
nsamples, ndim = flat_samples.shape
nsub = 500
idx = np.random.randint(0, nsamples, size=nsub)
subsample = flat_samples[idx]

gm = GetModel(emu_name, binning, iz, params_free_name, params_fixed_value, params_fixed_name)


data_vec = np.loadtxt(f'../data_vector/data_vector_{data_name}/data_vector_{rich_name}_{binning}_z{redshift}.dat')
cov = np.loadtxt(f'../data_vector/data_vector_abacus_summit/cov_z{redshift}.dat')


plt.figure(figsize=(14,7))
#### counts

if binning != 'abun':
    plt.subplot(1,2,1)
    x = np.arange(4)
    sigma = np.sqrt(np.diag(cov))
    plt.errorbar(x, data_vec[0:4], sigma[0:4], c='k', marker='o', mec='k', ls='', capsize=8)

    for i in range(nsub):
        plt.plot(x, gm.model(subsample[i])[0:4], c='gray', alpha=0.01)
    plt.xlim(-0.1,3.1)
    plt.yscale('log')
    plt.ylabel('Counts')
    plt.xlabel('richness bin')
    plt.title(f'{data_name} {rich_name} {binning} z={redshift}')


#### lensing
plt.subplot(1,2,2)
rp = np.loadtxt(train_loc+f'rp_rad.dat')
nrp = len(rp)

if binning == 'abun':
    lensing_vec = data_vec[:]
    lensing_sigma = np.sqrt(np.diag(cov))
else:
    lensing_vec = data_vec[4:]
    lensing_sigma = sigma[4:]

for ibin in range(4):
    plt.errorbar(rp, rp*lensing_vec[ibin*nrp:(ibin+1)*nrp], \
        rp*lensing_sigma[ibin*nrp:(ibin+1)*nrp],\
        c=f'C{ibin}', marker='o', mec=f'C{ibin}', ls='', capsize=8)

for i in range(nsub):
    if binning == 'abun':
        lensing_model = gm.model(subsample[i])
    else:
        lensing_model = gm.model(subsample[i])[4:]
    
    for ibin in range(4):
        plt.plot(rp, rp*lensing_model[ibin*nrp:(ibin+1)*nrp], c=f'C{ibin}', alpha=0.01)    
plt.xlim(0.9*min(rp), 1.1*max(rp))
plt.xscale('log')
plt.xlabel(r'$r_{\rm p}~[{\rm pMpc}]$')
plt.ylabel(r'$r_{\rm p} \Delta\Sigma~[{\rm pMpc ~M_\odot/pc^2} ]$')
plt.savefig(plot_loc+f'pred_{para_name}_z{redshift}_run{run_id}.pdf', dpi=72)
print(f'plot saved at {plot_loc}')
