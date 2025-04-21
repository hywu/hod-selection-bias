#!/usr/bin/env python
import fitsio
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import yaml
from hod.utils.read_sim import read_sim

## need 16G

class CalcGalDen(object):
    def __init__(self, yml_fname):

        with open(yml_fname, 'r') as stream:
            try:
                self.para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.depth = self.para['depth']
        perc = self.para['perc']
        
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
        else:
           output_loc = self.para['output_loc']

        #output_loc = self.para['output_loc']
        model_name = self.para['model_name']

        self.rich_name = self.para['rich_name'] #+ f'{self.depth}'
        self.out_path = f'{output_loc}/model_{model_name}'
        redshift = self.para['redshift']
        self.survey = self.para.get('survey', 'desy1')
        
        self.obs_path = f'{self.out_path}/obs_{self.rich_name}_{self.survey}/'
        #print('obs_path', self.obs_path)
        if os.path.isdir(self.obs_path)==False: 
            os.makedirs(self.obs_path)

        self.readcat = read_sim(self.para)

        #self.mpart = self.readcat.mpart
        self.boxsize = self.readcat.boxsize
        #self.hubble = self.readcat.hubble
        self.vol = self.boxsize**3

    def calc_gal_den(self):
        # calculate galaxy density
        den_fname = f'{self.out_path}/gal_density.dat'
        if os.path.exists(den_fname) == False:
            # read in galaxies
            gal_cat_format = self.para.get('gal_cat_format', 'fits')
            if gal_cat_format == 'fits':
                gal_fname = f'{self.out_path}/gals.fit'
                data, header = fitsio.read(gal_fname, header=True)
                x_gal_in = data['px']
            
            # if gal_cat_format == 'h5':
            #     import h5py
            #     loc = '/bsuhome/hwu/scratch/abacus_summit/'
            #     gal_fname = loc + 'NHOD_0.10_11.7_11.7_12.9_1.00_0.0_0.0_1.0_1.0_0.0_c000_ph000_z0p300.hdf5'
            #     f = h5py.File(gal_fname,'r')
            #     data = f['particles']
            #     #print(data.dtype)
            #     x_gal_in = data['x']

            ngal = len(x_gal_in)/self.vol
            data = np.array([ngal]).transpose()
            np.savetxt(den_fname, data, fmt='%-12g', header='ngal (h^3 Mpc^-3)')
        else:
            pass #print('gal density done')
        print('file saved', den_fname)

if __name__ == "__main__":
    yml_fname = sys.argv[1]
    cgd = CalcGalDen(yml_fname)
    cgd.calc_gal_den()

