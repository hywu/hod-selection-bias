#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import fitsio
import yaml
import sys

#from read_yml import ReadYML


#./plot_counts_richness.py yml/mini_uchuu_fid_hod.yml
#./plot_counts_richness.py yml/abacus_summit_fid_hod.yml

class PlotCountsRichness(object):
    def __init__(self, yml_fname): #, model_id, depth

        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        #self.model_id = model_id
        self.depth = para['depth']
        # self.out_path = out_path
        # self.vol = vol

        # with open(yml_fname, 'r') as stream:
        #     try:
        #         parsed_yaml = yaml.safe_load(stream)
        #     except yaml.YAMLError as exc:
        #         print(exc)
        # model_set = parsed_yaml['model_set']
        # nbody = parsed_yaml['nbody']
        # output_loc = parsed_yaml['output_loc']
        #para = ReadYML(yml_fname)
        #self.out_path = f'{para.output_loc}/model_{para.model_set}_{model_id}/'

        output_loc = para['output_loc']
        model_name = para['model_name']
        self.out_path = f'{output_loc}/model_{model_name}'

        if os.path.isdir(f'{self.out_path}/obs_d{self.depth:.0f}/')==False: 
            os.makedirs(f'{self.out_path}/obs_d{self.depth:.0f}/')
        '''
        if para.nbody == 'mini_uchuu':
            output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
            from read_mini_uchuu import ReadMiniUchuu
            rmu = ReadMiniUchuu()
            # self.xp, self.yp, self.zp = rmu.read_particles()
            # self.mpart = rmu.mpart
            self.boxsize = rmu.boxsize
            # self.hubble = rmu.hubble
            self.vol = self.boxsize**3

        # hod_name = f'model_{model_id}'
        # self.out_path = f'{output_loc}/{hod_name}/'
        '''
        if para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            self.readcat = ReadMiniUchuu(para['nbody_loc'])

        if para['nbody'] == 'abacus_summit':
            sys.path.append('../abacus_summit')
            from read_abacus_summit import ReadAbacusSummit
            self.readcat = ReadAbacusSummit(para['nbody_loc'])

        #self.mpart = self.readcat.mpart
        self.boxsize = self.readcat.boxsize
        #self.hubble = self.readcat.hubble
        self.vol = self.boxsize**3

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

        if plot_y1 == True: # TODO: This file is missing... OMG
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
    yml_fname = sys.argv[1]

    ccr = PlotCountsRichness(yml_fname)
    ccr.calc_counts_richness()
    ccr.plot_counts_richness(label='model0')
    plt.show()
