#!/usr/bin/env python
import numpy as np
from scipy import spatial
rmax_tree = 4
import yaml
import fitsio
from astropy.io import fits
import sys
from fid_hod import Ngal_S20_noscatt
from scipy.optimize import minimize


class PlotHOD(object):
    def __init__(self, yml_fname):
        with open(yml_fname, 'r') as stream:
            try:
                self.para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.redshift = self.para['redshift']
        self.mdef = self.para['mdef']
        #if self.mdef == 'R200m':
        #

        #### For AbacusSummit ####
        if self.para['nbody'] == 'abacus_summit':
            self.cosmo_id = self.para.get('cosmo_id', None)
            hod_id = self.para.get('hod_id', None)
            phase = self.para.get('phase', None)
            redshift = self.para['redshift']
            if redshift == 0.3: z_str = '0p300'
            output_loc = self.para['output_loc']+f'/base_c{self.cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'

        else:
           output_loc = self.para['output_loc']
           self.Om0 = self.para['OmegaM']

        #output_loc = self.para['output_loc']
        model_name = self.para['model_name']
        self.out_path = f'{output_loc}/model_{model_name}/'
        self.ofname1 = f'{self.out_path}/mass_Ngal.fit'
        self.ofname2 = f'{self.out_path}/hod_recovered.dat'
        self.ofname3 = f'{self.out_path}/hod_para_bestfit.yml'

    def calc_hod(self, Mmin):
        if self.para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            readcat = ReadMiniUchuu(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'uchuu':
            from read_uchuu import ReadUchuu
            readcat = ReadUchuu(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'abacus_summit':
            from read_abacus_summit import ReadAbacusSummit
            readcat = ReadAbacusSummit(self.para['nbody_loc'], self.redshift, self.cosmo_id)
            self.Om0 = readcat.OmegaM

        if self.para['nbody'] == 'flamingo':
            from read_flamingo import ReadFlamingo
            readcat = ReadFlamingo(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'tng_dmo':
            from read_tng_dmo import ReadTNGDMO
            halofinder = para.get('halofinder', 'rockstar')
            readcat = ReadTNGDMO(self.para['nbody_loc'], halofinder, self.redshift)
            print('halofinder', halofinder)

       
        readcat.read_halos(Mmin)
        mass = readcat.mass
        id_halo = readcat.hid
        x_halo = readcat.xh
        y_halo = readcat.yh
        z_halo = readcat.zh

        gal_fname = f'{self.out_path}/gals.fit'
        data, header = fitsio.read(gal_fname, header=True)

        x_gal = data['px']
        y_gal = data['py']
        z_gal = data['pz']

        gal_position = np.dstack([x_gal, y_gal, z_gal])[0]
        gal_tree = spatial.cKDTree(gal_position)

        halo_position = np.dstack([x_halo, y_halo, z_halo])[0]
        halo_tree = spatial.cKDTree(halo_position)
        indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)

        rhocrit = 2.775e11 # h-unit
        if self.mdef == '200m':
            radius = (3 * mass / (4 * np.pi * rhocrit * self.Om0 * 200.))**(1./3.) # cMpc/h (chimp)
        elif self.mdef == 'vir':
            OmegaM_z = self.Om0 * (1+ self.redshift)**3 / (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)
            x = OmegaM_z - 1
            Delta_vir_c = 18 * np.pi**2 + 82 * x - 39 * x**2
            rhocrit_z = rhocrit * (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)/(1+self.redshift)**3 # gotcha!
            radius = (3 * mass / (4 * np.pi * rhocrit_z * Delta_vir_c))**(1./3.)

        nhalo = len(x_halo)
        ngal = []
        for i_halo in range(nhalo):
            gal_ind = indexes_tree[i_halo]
            x_cen = x_halo[i_halo]
            y_cen = y_halo[i_halo]
            z_cen = z_halo[i_halo]
            #r = (x_gal[gal_ind] - x_cen)**2 + (y_gal[gal_ind] - y_cen)**2 + (z_gal[gal_ind] - z_cen)**2 
            #r = np.sqrt(r)
            #ngal.append(len(r[r <= radius[i_halo]*(1+1e-4)]))
            indx = gal_tree.query_ball_point([x_cen, y_cen, z_cen], radius[i_halo])
            ngal.append(len(indx))
        ngal = np.array(ngal)

        #### save the Ngals ####
        col0 = fits.Column(name='haloid', unit='', format='D', array=id_halo)
        col1 = fits.Column(name='mass', unit='', format='D', array=mass)
        col2 = fits.Column(name='Ngal', unit='', format='D', array=ngal)
        cols = [col0, col1, col2]
        coldefs = fits.ColDefs(cols)
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.writeto(self.ofname1, overwrite=True)

        #### calculate the average ####
        x = np.log(mass)
        y = ngal

        nbins_per_decade = 10
        n_decade = (np.log10(max(mass))-np.log10(min(mass)))
        nbins = int(nbins_per_decade*n_decade + 1e-4) 
        x_bins = np.linspace(min(x), max(x), nbins+1)
        x_bin_mean = []
        y_bin_mean = []
        y_bin_std = []
        for i in range(nbins):
            sel = (x > x_bins[i])&(x < x_bins[i+1])
            x_bin_mean.append(np.mean(x[sel]))
            y_bin_mean.append(np.mean(y[sel]))
            y_bin_std.append(np.std(y[sel]))

        y_bin_mean = np.array(y_bin_mean)
        y_bin_std = np.array(y_bin_std)

        data = np.array([np.exp(x_bin_mean), y_bin_mean, y_bin_std]).transpose()
        np.savetxt(self.ofname2, data, fmt='%-12g', header='mass, mean(Ngal), std(N_gal)')

        return np.exp(x_bin_mean), y_bin_mean, y_bin_std


    def plot_hod(self):
        m, mean_Ngal, std_ngal = np.loadtxt(self.ofname2, unpack=True)
        return m, mean_Ngal, std_ngal

    def plot_Ngal_vs_mass(self):
        data, h = fitsio.read(self.ofname1, header=True)
        m = data['mass']
        Ngal = data['Ngal']
        return m, Ngal


    def fit_hod(self):
        mass, hod_mean, hod_std = self.plot_hod()
        sel = (hod_mean > 1e-2)
        mass = mass[sel]
        hod_mean = hod_mean[sel]
        hod_std = hod_std[sel]

        def chi_sqr(para):
            alpha = para[0]
            lgM1 = para[1]
            kappa = para[2]
            lgMcut= para[3]
            sigmalogM = para[4]
            Ncen, Nsat = Ngal_S20_noscatt(mass, alpha, lgM1, kappa, lgMcut, sigmalogM)
            return np.sum((Ncen + Nsat - hod_mean)**2/hod_std**2)

        alpha_ini = 1
        lgM1_ini = 13.5
        kappa_ini = 1
        lgMcut_ini = 12
        sigmalogM_ini = 0.1
        res = minimize(chi_sqr, x0=(alpha_ini, lgM1_ini, kappa_ini, lgMcut_ini, sigmalogM_ini), 
            bounds=((0.5, 1.5), (13, 14), (0.5,2), (11, 13), (0.05, 0.5)))
        print(res)
        alpha_best = res.x[0]
        lgM1_best = res.x[1]
        kappa_best = res.x[2]
        lgMcut_best = res.x[3]
        sigmalogM_best = res.x[4]
        # lgMcut_best = res.x[0]
        # lgM1_best = res.x[1]
        # alpha_best = res.x[2]
        # sigmalogM_best = res.x[3]
        # kappa_best = res.x[4]
        Ncen_best, Nsat_best = Ngal_S20_noscatt(mass, alpha_best, lgM1_best, kappa_best, lgMcut_best, sigmalogM_best)

        para_dict = {
        "lgMcut": float(lgMcut_best),
        "lgM1": float(lgM1_best),
        "alpha": float(alpha_best),
        "sigmalogM": float(sigmalogM_best), 
        "kappa": float(kappa_best),
        }
        with open(self.ofname3, 'w') as outfile:
            yaml.dump(para_dict, outfile)

        return para_dict

    def plot_hod_bestfit(self):
        mass, hod_mean, hod_std = self.plot_hod()
        sel = (hod_mean > 1e-2)
        mass = mass[sel]

        with open(self.ofname3, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
                print(parsed_yaml)
            except yaml.YAMLError as exc:
                print(exc)
        alpha = parsed_yaml['alpha']
        lgM1 = parsed_yaml['lgM1']
        kappa = parsed_yaml['kappa']
        lgMcut = parsed_yaml['lgMcut']
        sigmalogM = parsed_yaml['sigmalogM']
        Ncen, Nsat = Ngal_S20_noscatt(mass, alpha, lgM1, kappa, lgMcut, sigmalogM)
        return mass, Ncen, Nsat


if __name__ == "__main__":
    yml_fname = sys.argv[1] 
    #./plot_hod.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml

    ch = PlotHOD(yml_fname)
    mass, hod_mean, hod_std = ch.calc_hod(Mmin=1e15)

    
    # use plot_hod_mor.ipynb for plotting
    '''
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    plt.loglog(mass, hod_mean, 'o-')

    plt.subplot(122)
    plt.semilogx(mass, hod_std)

    plt.show()
    '''