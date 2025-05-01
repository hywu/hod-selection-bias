#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fitsio
import os
import sys
import pandas as pd
#sys.path.append('../utils')
from hod.utils.get_para_abacus_summit import get_cosmo_para

class ReadAbacusSummit(object):
    def __init__(self, nbody_loc, redshift, cosmo_id, phase):

        cosmo_para = get_cosmo_para(cosmo_id)
        self.hubble = cosmo_para['hubble']
        self.OmegaM = cosmo_para['OmegaM']

        if redshift == 0.3: z_str = '0p300'
        if redshift == 0.4: z_str = '0p400'
        if redshift == 0.5: z_str = '0p500'

        self.input_loc = nbody_loc + f'/base_c{cosmo_id:0>3d}/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'

        self.boxsize = 2000.
        rhocrit = 2.77536627e11 #h^-1 M_sun Mpc^-3
        self.part_fname = self.input_loc + f'subsample_particles_A_base_c{cosmo_id:0>3d}_ph{phase:0>3d}_z{z_str}.h5'
        f = h5py.File(self.part_fname, 'r')
        particles = f['particles']
        self.npart = np.shape(particles)[0]
        self.mpart = self.OmegaM * rhocrit * self.boxsize**3 / self.npart

        self.halo_fname_original = self.input_loc + f'halo_base_c{cosmo_id:0>3d}_ph{phase:0>3d}_z{z_str}.h5' 

    def read_halos(self, Mmin=1e11, pec_vel=False):

        if Mmin >= 2.99e+12:
            halo_fname = self.input_loc+f'halos_3e+12.fit'
        elif Mmin >= 9.99e+10 and Mmin < 2.99e+12:
            halo_fname = self.input_loc+f'halos_1e+11.fit'
        else:
            halo_fname = 'dummy.fit'

        if os.path.exists(halo_fname):  # if I have saved the file.
            print('halo file', halo_fname)
            fsize = os.path.getsize(halo_fname)/1024**3
            print(f'halo file size {fsize:.2g} GB')
            data = fitsio.read(halo_fname)
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

        else: 
            print(self.halo_fname_original)
            f = h5py.File(self.halo_fname_original,'r')
            halos = f['halos']
            #print(halos.dtype)
            mass = halos['mass']
            x_halo = halos['x']
            y_halo = halos['y']
            z_halo = halos['z']
            vx_halo = halos['vx']
            vy_halo = halos['vy']
            vz_halo = halos['vz']
            hid = halos['gid']

            sel = (mass > Mmin)
            x_halo_sel = x_halo[sel]
            y_halo_sel = y_halo[sel]
            z_halo_sel = z_halo[sel]
            vx_halo_sel = vx_halo[sel]
            vy_halo_sel = vy_halo[sel]
            vz_halo_sel = vz_halo[sel]
            mass_sel = mass[sel]
            hid_sel = hid[sel]

            index = np.argsort(-mass_sel)
            self.xh = x_halo_sel[index]
            self.yh = y_halo_sel[index]
            self.zh = z_halo_sel[index]
            self.vx = vx_halo_sel[index]
            self.vy = vy_halo_sel[index]
            self.vz = vz_halo_sel[index]
            self.mass = mass_sel[index]
            self.hid = hid_sel[index]

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
    ras = ReadAbacusSummit(nbody_loc, redshift=0.4, cosmo_id=0, phase=0)
    ras.read_halos(Mmin=3e12)
    print(len(ras.xh))
    plt.scatter(ras.xh[ras.zh<10], ras.yh[ras.zh<10], s=0.1)
    plt.savefig('test.png')

    #x, y, z = ras.read_particle_positions()
    # plt.figure(figsize=(10,10))
    # plt.scatter(ras.xp[ras.zp<10], ras.yp[ras.zp<10], s=0.1)
    # plt.savefig('test2.png')
    
    #vx, vy, vz = ras.read_particle_velocities() ## doesn't have it

