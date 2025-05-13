#!/usr/bin/env python
import fitsio
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml
from hod.utils.read_sim import read_sim
from astropy.cosmology import FlatLambdaCDM

#sys.path.append('../utils')

#./plot_abundance.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml
#./plot_abundance.py ../yml/abacus_summit/abacus_summit_template.yml

#### comoving density * comoving volume => observed counts

class PlotAbundance(object):
    def __init__(self, yml_fname, zmin, zmax, survey_area_sq_deg):

        with open(yml_fname, 'r') as stream:
            try:
                self.para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        
        #### For AbacusSummit ####
        if self.para['nbody'] == 'abacus_summit':
            cosmo_id = self.para.get('cosmo_id', None)
            hod_id = self.para.get('hod_id', None)
            phase = self.para.get('phase', None)
            redshift = self.para['redshift']
            if redshift == 0.3: z_str = '0p300'
            if redshift == 0.4: z_str = '0p400'
            if redshift == 0.5: z_str = '0p500'
            output_loc = self.para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
            from hod.utils.get_para_abacus_summit import get_cosmo_para, get_hod_para
            cosmo_abacus = get_cosmo_para(cosmo_id)
            #h = cosmo_abacus['hubble']
            Om0 = cosmo_abacus['OmegaM']

            self.binning = self.para.get('binning', 'Ncyl')
            if self.binning == 'AB_scaling':
                hod_para = get_hod_para(hod_id)
                self.A = hod_para['A']
                self.B = hod_para['B']

        else:
           output_loc = self.para['output_loc']

        model_name = self.para['model_name']

        self.rich_name = self.para['rich_name']
        self.out_path = f'{output_loc}/model_{model_name}'
        redshift = self.para['redshift']
        self.survey = self.para.get('survey', 'desy1')

        self.obs_path = f'{self.out_path}/obs_{self.rich_name}_{self.survey}/'

        if os.path.isdir(self.obs_path)==False: 
            os.makedirs(self.obs_path)

        self.readcat = read_sim(self.para)


        self.boxsize = self.readcat.boxsize
        Om0 = self.readcat.OmegaM
        self.sim_vol = self.boxsize**3
        #print('sim_vol', self.sim_vol)

        self.ofname = f'{self.obs_path}/abundance.dat'
        if self.binning == 'AB_scaling':
            self.ofname = f'{self.obs_path}/abundance_AB.dat'
        print('abundance saved at:', self.ofname)

        fsky = survey_area_sq_deg/41253.
        cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
        
        self.survey_vol = fsky * 4. * np.pi/3. * (cosmo.comoving_distance(zmax).value**3 - cosmo.comoving_distance(zmin).value**3) #* h**3  # (h/Mpc)**3
        #print('sim_vol', self.sim_vol)
        #print('survey_vol', self.survey_vol)


    def calc_abundance(self, rich_fname=None): # allow intput fits file name directly
        lambda_min_list = [ 20, 30, 45, 60]
        lambda_max_list = [ 30, 45, 60, 1000]

        if rich_fname == None:
            rich_fname = f'{self.out_path}/richness_{self.rich_name}.fit'

        data = fitsio.read(rich_fname)
        lam = data['lambda']

        if self.binning == 'AB_scaling':
            print('AB_scaling', self.A, self.B)
            lnlam_new = self.A * np.log(lam) + self.B
            lam = np.exp(lnlam_new)

        counts_list = []
        for ilam in range(len(lambda_min_list)):
            sel = (lam >= lambda_min_list[ilam]) * (lam < lambda_max_list[ilam])
            counts_list.append(len(lam[sel]) / self.sim_vol * self.survey_vol)

        data = np.array([lambda_min_list, lambda_max_list, counts_list]).transpose()
        np.savetxt(self.ofname, data, fmt='%-12g', header='lam_min, lam_max, counts')
        #print('counts = ', counts_list)

if __name__ == "__main__":
    yml_fname = sys.argv[1]
    ccr = PlotAbundance(yml_fname, zmin=0.2, zmax=0.35, survey_area_sq_deg=1437)
    ccr.calc_abundance()

