#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import fitsio

class CalcCountsRichness(object):
    def __init__(self, which_sim, model_id, depth):
        self.model_id = model_id
        self.depth = depth
        # self.out_path = out_path
        # self.vol = vol

        if which_sim == 'mini_uchuu':
            output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
            from read_mini_uchuu import ReadMiniUchuu
            rmu = ReadMiniUchuu()
            # self.xp, self.yp, self.zp = rmu.read_particles()
            # self.mpart = rmu.mpart
            self.boxsize = rmu.boxsize
            # self.hubble = rmu.hubble
            self.vol = self.boxsize**3

        hod_name = f'model_{model_id}'
        self.out_path = f'{output_loc}/{hod_name}/'

        self.ofname = f'{self.out_path}/obs_d{self.depth:.0f}/counts_richness.dat'

    def calc_counts_richness(self):
        if self.depth == 'pmem' or self.depth==-1:
            fname = f'{self.out_path}/richness_pmem.fit'
        else:
            fname = f'{self.out_path}/richness_d{self.depth:.0f}.fit'

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
            gal_fname = f'{self.out_path}/gals.fit'
            data_gal = fitsio.read(gal_fname)
            ngal = len(data_gal['px'])/self.vol
            data = np.array([ngal]).transpose()
            np.savetxt(fname, data, fmt='%-12g', header='ngal (h^3 Mpc^-3)')

        data = np.array([lam_min_list, den_list]).transpose()
        np.savetxt(self.ofname, data, fmt='%-12g', header='lam_min, den')

    def plot_counts_richness(self, axes=None, label=None, plot_y1=False):
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(7, 7))

        if plot_y1 == True:
            lam_min_list, den_list, den_low, den_high = np.loadtxt('../y1/data/des_y1_space_density_lambda_z_0.2_0.35.dat', unpack=True)
            plt.plot(lam_min_list, den_list, label='DES Y1', c='k')
            plt.fill_between(lam_min_list, den_low, den_high, facecolor='gray', alpha=0.2)
        
        if label is None: label = ''
            
        ngal = np.loadtxt(f'{self.out_path}/gal_density.dat')
        label += r'$, \rm n_{gal}$=%.2e'%(ngal)

        lam_min_list, den_list = np.loadtxt(self.ofname, unpack=True)
        plt.loglog(lam_min_list, den_list, label=label)
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$n(>\lambda)$')
        plt.legend()
        plt.xlim(10, None)


if __name__ == "__main__":
    model_id = 0
    depth = 30
    # output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
    # out_path = f'{output_loc}/model_{model_id}/'
    # vol = 400**3
    ccr = CalcCountsRichness('mini_uchuu', model_id, depth, vol)
    ccr.calc_counts_richness()
    ccr.plot_counts_richness(label='model0')
    plt.show()
