#!/usr/bin/env python
import numpy as np
import h5py, fitsio
import os, sys
sys.path.append('../utils')
from readGadgetSnapshot import readGadgetSnapshot

class ReadMiniUchuu(object):
    def __init__(self, nbody_loc, redshift):
        self.input_loc = nbody_loc #'/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
        #snap_header = readGadgetSnapshot(self.input_loc+f'snapdir_043/MiniUchuu_043.gad.0') # the file is too big to copy

        self.hubble = 0.6774 #snap_header.HubbleParam 
        self.OmegaM = 0.3089 #snap_header.Omega0
        self.w0 = -1
        self.wa = 0
        self.boxsize = 400. #snap_header.BoxSize
        self.mpart = 3.270422e+11 #snap_header.mass[1] * 1e10 * 1000
        self.redshift = redshift

        if self.redshift == 0.3:
            self.snap_name = '043'
        if self.redshift == 0.1:
            self.snap_name = '047'

    def read_halos(self, Mmin=1e11, pec_vel=False, cluster_only=False):

        data = fitsio.read(self.input_loc+f'host_halos_{self.snap_name}.fit')
        M200m = data['M200m']
        sel = (M200m >= Mmin)
        M200m = M200m[sel]
        sort = np.argsort(-M200m)
        self.mass = M200m[sort]

        self.hid = data['haloid'][sel][sort]
        self.xh = data['px'][sel][sort]
        self.yh = data['py'][sel][sort]
        self.zh = data['pz'][sel][sort]

        if pec_vel == True:
            self.vx = data['vx'][sel][sort]
            self.vy = data['vy'][sel][sort]
            self.vz = data['vz'][sel][sort]

    def read_particle_positions(self):
        data = fitsio.read(self.input_loc+f'particles_{self.snap_name}_0.1percent.fit')
        self.xp = data['x']
        self.yp = data['y']
        self.zp = data['z']
        # self.vxp = data['vx']
        # self.vyp = data['vy']
        # self.vzp = data['vz']
        return self.xp, self.yp, self.zp

    def read_particle_velocities(self):
        data = fitsio.read(self.input_loc+f'particles_{self.snap_name}_0.1percent.fit')
        self.vxp = data['vx']
        self.vyp = data['vy']
        self.vzp = data['vz']
        return self.vxp, self.vyp, self.vzp

    def read_particles_layer(self, pz_min, pz_max):
        fname = self.input_loc+f'layers_{self.snap_name}/particles_pz_{pz_min}_{pz_max}_0.1percent.h5'
        f = h5py.File(fname, 'r')
        data = f['part']
        self.xp = data['x']
        self.yp = data['y']
        self.zp = data['z']
        
        self.vxp = data['vx']
        self.vyp = data['vy']
        self.vzp = data['vz']

        return self.xp, self.yp, self.zp, self.vxp, self.vyp, self.vzp


if __name__ == '__main__':
    rmu = ReadMiniUchuu(nbody_loc='/bsuhome/hwu/scratch/uchuu/MiniUchuu/', redshift=0.1)
    # rmu.read_halos(pec_vel=True)
    # print('max', np.max(rmu.vx))
    # print('mean, std', np.mean(rmu.vx), np.std(rmu.vx))
    rmu.read_particles_layer(0.0, 4.0)
    print(len(rmu.xp), len(rmu.vxp))
