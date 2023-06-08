#!/usr/bin/env python
import numpy as np
import yaml
import fitsio

def plot_mor(yml_fname):
    with open(yml_fname, 'r') as stream:
        try:
            para = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    output_loc = para['output_loc']
    model_name = para['model_name']
    out_path = f'{output_loc}/model_{model_name}/'

    use_pmem = para.get('use_pmem', False)
    depth = para['depth']
    rich_name = para['rich_name']
    survey = para.get('survey', 'desy1')

    cl_fname = f'{out_path}/richness_{rich_name}.fit'
    print(cl_fname)
    data = fitsio.read(cl_fname)
    mass = data['M200m']
    lam = data['lambda']

    x = np.array(np.log(mass[lam > 0]))
    y = np.array(np.log(lam[lam > 0]))

    nbins_per_decade = 5
    n_decade = (np.log10(max(mass))-np.log10(min(mass)))
    nbins = int(nbins_per_decade*n_decade + 1e-4) 
    x_bins = np.linspace(min(x), max(x), nbins+1)

    x_bin_mean = []
    y_bin_mean = []
    y_bin_scat = []
    for i in range(nbins):
        sel = (x > x_bins[i])&(x < x_bins[i+1])
        x_bin_mean.append(np.mean(x[sel]))
        y_bin_mean.append(np.mean(y[sel]))
        y_bin_scat.append(np.std(y[sel]))

    ofname = f'{out_path}/obs_{rich_name}_{survey}/mor.dat'
    data = np.array([np.exp(x_bin_mean), np.exp(y_bin_mean), y_bin_scat]).transpose()
    np.savetxt(ofname, data, fmt='%-12g', header='m, exp<lnlam>, std(lnlam)')
    return np.exp(x_bin_mean), np.exp(y_bin_mean), y_bin_scat



