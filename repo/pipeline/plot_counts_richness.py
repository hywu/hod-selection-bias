#!/usr/bin/env python
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml
sys.path.append('../utils')

#./plot_counts_richness.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml
#./plot_counts_richness.py yml/abacus_summit_fid_hod.yml

class PlotCountsRichness(object):
    def __init__(self, yml_fname):

        with open(yml_fname, 'r') as stream:
            try:
                self.para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.depth = self.para['depth']
        perc = self.para['perc']
        output_loc = self.para['output_loc']
        model_name = self.para['model_name']
        self.rich_name = self.para['rich_name']
        self.out_path = f'{output_loc}/model_{model_name}'
        redshift = self.para['redshift']
        los = self.para.get('los', 'z')
        # if los == 'xyz':
        #     self.los = sys.argv[2]
        # else:
        #     self.los = los

        #use_pmem = self.para.get('use_pmem', False)
        #pec_vel = self.para.get('pec_vel', False)
        #sat_from_part = self.para.get('sat_from_part', False)

        #if use_pmem == True:
        #    self.rich_name = f'pmem'
        #else:
        #    self.rich_name = f'd{self.depth:.0f}'

        #if self.los == 'x' or self.los == 'y':
        #    self.rich_name = f'{self.rich_name}_{self.los}'

        #if pec_vel == True:
        #    self.rich_name += '_vel'
        #if perc == False:
        #    self.rich_name += '_noperc'
        #if sat_from_part == True:
        #    self.rich_name += '_from_part'


        if os.path.isdir(f'{self.out_path}/obs_{self.rich_name}/')==False: 
            os.makedirs(f'{self.out_path}/obs_{self.rich_name}/')

        if self.para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            self.readcat = ReadMiniUchuu(self.para['nbody_loc'], redshift)

        if self.para['nbody'] == 'abacus_summit':
            sys.path.append('../abacus_summit')
            from read_abacus_summit import ReadAbacusSummit
            self.readcat = ReadAbacusSummit(self.para['nbody_loc'], redshift)

        if self.para['nbody'] == 'tng_dmo':
            from read_tng_dmo import ReadTNGDMO
            halofinder = self.para.get('halofinder', 'rockstar')
            self.readcat = ReadTNGDMO(self.para['nbody_loc'], halofinder, redshift)
            print('halofinder', halofinder)

        #self.mpart = self.readcat.mpart
        self.boxsize = self.readcat.boxsize
        #self.hubble = self.readcat.hubble
        self.vol = self.boxsize**3

        self.ofname = f'{self.out_path}/obs_{self.rich_name}/counts_richness.dat'

    def calc_counts_richness(self):
        if self.depth == 'pmem' or self.depth==-1:
            fname = f'{self.out_path}/richness_{self.rich_name}.fit'
        else:
            fname = f'{self.out_path}/richness_{self.rich_name}.fit'

        #print('self.rich_name', self.rich_name)
        data = fitsio.read(fname)
        lam = data['lambda']
        lam_min_list = 10**np.linspace(np.log10(20), np.log10(100), 20)

        den_list = []
        for lam_min in lam_min_list:
            sel = (lam >= lam_min)
            den_list.append(len(lam[sel])/self.vol)

        # get galaxy density
        fname = f'{self.out_path}/gal_density.dat'
        if os.path.exists(fname) == False:
            
            # read in galaxies
            gal_cat_format = self.para.get('gal_cat_format', 'fits')

            if gal_cat_format == 'fits':
                gal_fname = f'{self.out_path}/gals.fit'
                data, header = fitsio.read(gal_fname, header=True)
                x_gal_in = data['px']

            if gal_cat_format == 'h5':
                import h5py
                loc = '/bsuhome/hwu/scratch/abacus_summit/'
                gal_fname = loc + 'NHOD_0.10_11.7_11.7_12.9_1.00_0.0_0.0_1.0_1.0_0.0_c000_ph000_z0p300.hdf5'
                f = h5py.File(gal_fname,'r')
                data = f['particles']
                #print(data.dtype)
                x_gal_in = data['x']

            ngal = len(x_gal_in)/self.vol
            data = np.array([ngal]).transpose()
            np.savetxt(fname, data, fmt='%-12g', header='ngal (h^3 Mpc^-3)')

        data = np.array([lam_min_list, den_list]).transpose()
        np.savetxt(self.ofname, data, fmt='%-12g', header='lam_min, den')

    def plot_counts_richness(self, axes=None, label=None):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(7, 7))
        if label is None: 
            label = ''
            
        ngal = np.loadtxt(f'{self.out_path}/gal_density.dat')
        label += r'$, \rm n_{gal}$=%.2e'%(ngal)

        lam_min_list, den_list = np.loadtxt(self.ofname, unpack=True)
        plt.loglog(lam_min_list, den_list, label=label)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$n(>\lambda)$')
        plt.legend()
        plt.xlim(10, None)

    def plot_y1_counts_richness(self):
        lam_min_list, den_list, den_low, den_high = np.loadtxt('../y1/data/des_y1_space_density_lambda_z_0.2_0.35.dat', unpack=True)
        plt.plot(lam_min_list, den_list, label='DES Y1', c='k')
        plt.fill_between(lam_min_list, den_low, den_high, facecolor='gray', alpha=0.2)


if __name__ == "__main__":
    yml_fname = sys.argv[1]

    ccr = PlotCountsRichness(yml_fname)
    ccr.calc_counts_richness()
    ccr.plot_counts_richness()
    plt.show()
