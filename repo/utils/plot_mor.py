#!/usr/bin/env python
import numpy as np
import yaml
import fitsio
import sys

def plot_mor(yml_fname, rich_fname=None, exclude_zero=True, save_file=False):

    with open(yml_fname, 'r') as stream:
        try:
            para = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    #### For AbacusSummit ####
    if para['nbody'] == 'abacus_summit':
        cosmo_id = para.get('cosmo_id', None)
        hod_id = para.get('hod_id', None)
        phase = para.get('phase', None)
        redshift = para['redshift']
        if redshift == 0.3: z_str = '0p300'
        output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
    else:
       output_loc = para['output_loc']

    #output_loc = para['output_loc']
    
    model_name = para['model_name']
    out_path = f'{output_loc}/model_{model_name}/'

    use_pmem = para.get('use_pmem', False)
    depth = para['depth']
    rich_name = para['rich_name']
    survey = para.get('survey', 'desy1')

    if rich_fname == None:
        rich_fname = f'{out_path}/richness_{rich_name}.fit'
    #print(rich_fname)

    # otherwise, use input rich_name directly
    #print(rich_fname)
    data = fitsio.read(rich_fname)
    mass = data['mass_host'] #data['M200m']
    lam = data['lambda']

    if exclude_zero == True: 
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

        if save_file == True:
            ofname = f'{out_path}/obs_{rich_name}_{survey}/mor.dat'
            data = np.array([np.exp(x_bin_mean), np.exp(y_bin_mean), y_bin_scat]).transpose()
            np.savetxt(ofname, data, fmt='%-12g', header='m, exp<lnlam>, std(lnlam)')
        return np.exp(x_bin_mean), np.exp(y_bin_mean), y_bin_scat


    if exclude_zero == False:
        # allowing zero or negative lambda! 
        # doesn't work with background subtraction
        x = np.array(np.log(mass))
        y = np.array(lam)

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
            y_bin_scat.append(np.std(y[sel])/np.mean(y[sel]))

        return np.exp(x_bin_mean), y_bin_mean, y_bin_scat


if __name__ == "__main__":
    yml_fname = sys.argv[1] 
    plot_mor(yml_fname, save_file=True)
