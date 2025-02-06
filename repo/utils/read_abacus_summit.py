#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fitsio
import os
import sys
import pandas as pd
#sys.path.append('../utils')
#from readGadgetSnapshot import readGadgetSnapshot



#### cosmo_c***_ph***_z0p***.param files seem unreliable. 
#### read directly from the csv file
def get_cosmo_para(cosmo_id_wanted):
    df = pd.read_csv('/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/cosmologies.csv', sep=',')
    df.columns = df.columns.str.replace(' ', '')
    #print(df.columns)
    nrows = df.shape[0]
    for irow in np.arange(nrows):
        # retrieve one cosmology at a time
        row = df.iloc[irow]
        root = row['root'].replace(' ', '')
        cosmo_id = int(root[-3:])

        if cosmo_id == cosmo_id_wanted:
            print('cosmo_id', cosmo_id)
            hubble = row['h']
            OmegaB = row['omega_b']/hubble**2
            if len(row['omega_ncdm']) == 12:
                Oncdm = float(row['omega_ncdm'])/hubble**2
            else:
                Oncdm = float(row['omega_ncdm'][0:10])/hubble**2

            OmegaM = row['omega_cdm']/hubble**2 + OmegaB + Oncdm
            OmegaL = 1 - OmegaM
            sigma8 = row['sigma8_cb'] # baryons-plus-cdm-only  (CLASS)
            ns = row['n_s']
            break

    cosmo_dict = {'OmegaM': OmegaM, 'OmegaL': OmegaL,'hubble': hubble,'sigma8': sigma8,'OmegaB': OmegaB,'ns': ns}
    return cosmo_dict


class ReadAbacusSummit(object):
    def __init__(self, nbody_loc, redshift, cosmo_id, phase=0): #):

        cosmo_para = get_cosmo_para(cosmo_id)
        self.hubble = cosmo_para['hubble']
        self.OmegaM = cosmo_para['OmegaM']
        #print(cosmo_para)

        if redshift == 0.3: z_str = '0p300'

        self.input_loc = nbody_loc + f'/base_c{cosmo_id:0>3d}/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'

        self.boxsize = 2000
        rhocrit = 2.77536627e11 #h^-1 M_sun Mpc^-3
        self.part_fname = self.input_loc + f'subsample_particles_A_base_c{cosmo_id:0>3d}_ph{phase:0>3d}_z{z_str}.h5'
        f = h5py.File(self.part_fname, 'r')
        particles = f['particles']
        npart = np.shape(particles)[0]
        self.mpart = self.OmegaM * rhocrit * self.boxsize**3 / npart
        #print('mpart%e'%self.mpart)
        #exit()
        #self.input_loc = '/bsuhome/hwu/scratch/abacus_summit/'
        #snap_header = self.input_loc+'header'
        # self.hubble = 0.6736
        # self.OmegaM = 0.31519
        # self.boxsize = 2e3
        # self.mpart = 7.030000e+11#2.109e+09 / 0.003

    def read_halos(self, Mmin=1e11, pec_vel=False):
        # if Mmin < 3e12:
        #     halo_fname = self.input_loc + 'halo_base_c000_ph000_z0p300.h5'
        #     print(halo_fname)
        #     f = h5py.File(halo_fname,'r')
        #     halos = f['halos']
        #     mass = halos['mass']
        #     sel = (mass > Mmin)
        #     mass = mass[sel]
        #     hid = halos['gid'][sel]
        #     xh = halos['x'][sel]
        #     yh = halos['y'][sel]
        #     zh = halos['z'][sel]
        # else:
        #halo_fname = self.input_loc+f'halos_{Mmin:.0e}.fit'
        if Mmin <= 2.99e12:
            halo_fname = self.input_loc+f'halos_1e+11.fit'
        else: 
            halo_fname = self.input_loc+f'halos_3e+12.fit'

        print('halo file', halo_fname)
        fsize = os.path.getsize(halo_fname)/1024**3
        print(f'halo file size {fsize:.2g} GB')
        data = fitsio.read(halo_fname)
        # mass = data['mass']
        # sel = (mass > Mmin)
        # mass = mass[sel]
        # hid = data['haloid'][sel]
        # xh = data['px'][sel]
        # yh = data['py'][sel]
        # zh = data['pz'][sel]

        # sort = np.argsort(-mass)
        # self.mass = mass[sort]
        # self.hid = hid[sort]
        # self.xh = xh[sort]
        # self.yh = yh[sort]
        # self.zh = zh[sort]
        # #return self.hid, self.mass, self.xh, self.yh, self.zh
        mass = data['mass']
        sel = (mass >= Mmin)
        mass = mass[sel]
        sort = np.argsort(-mass)
        self.mass = mass[sort]

        self.hid = data['haloid'][sel][sort]
        self.xh = data['px'][sel][sort]
        self.yh = data['py'][sel][sort]
        self.zh = data['pz'][sel][sort]

        if pec_vel == True:
            self.vx = data['vx'][sel][sort]
            self.vy = data['vy'][sel][sort]
            self.vz = data['vz'][sel][sort]

    # def read_particles(self, pec_vel=False): #small enough to read both
    #     self.part_fname = self.input_loc + 'subsample_particles_A_base_c000_ph000_z0p300.h5'
    #     f = h5py.File(self.part_fname, 'r')
    #     particles = f['particles']
    #     self.xp = particles['x']
    #     self.yp = particles['y']
    #     self.zp = particles['z']
    #     self.vxp = data['vx']
    #     self.vyp = data['vy']
    #     self.vzp = data['vz']
    #     # return self.xp, self.yp, self.zp


    def read_particle_positions(self):
        #self.part_fname = self.input_loc + 'subsample_particles_A_base_c000_ph000_z0p300.h5'
        f = h5py.File(self.part_fname, 'r')
        particles = f['particles']
        self.xp = particles['x']
        self.yp = particles['y']
        self.zp = particles['z']
        return self.xp, self.yp, self.zp

    def read_particle_velocities(self):
        pass
        # Some files don't have particle velocities
        # self.part_fname = self.input_loc + 'subsample_particles_A_base_c000_ph000_z0p300.h5'
        # f = h5py.File(self.part_fname, 'r')
        # data = f['particles']
        # print(data.dtype)
        # self.vxp = data['vx']
        # self.vyp = data['vy']
        # self.vzp = data['vz']
        # return self.vxp, self.vyp, self.vzp


if __name__ == '__main__':
    
    nbody_loc = '/projects/hywu/cluster_sims/cluster_finding/data/AbacusSummit_base/'

    ras = ReadAbacusSummit(nbody_loc, redshift=0.3)
    #ras.read_halos(Mmin=3e12)
    # print(len(ras.xh))
    # plt.scatter(ras.xh[ras.zh<10], ras.yh[ras.zh<10], s=0.1)
    # plt.savefig('test.png')

    x, y, z = ras.read_particle_positions()
    # plt.figure(figsize=(10,10))
    # plt.scatter(ras.xp[ras.zp<10], ras.yp[ras.zp<10], s=0.1)
    # plt.savefig('test2.png')
    
    #vx, vy, vz = ras.read_particle_velocities() ## doesn't have it

