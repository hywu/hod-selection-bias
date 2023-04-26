#!/usr/bin/env python
import numpy as np
import h5py, fitsio
import os, sys
sys.path.append('../utils')
from readGadgetSnapshot import readGadgetSnapshot

class ReadUchuu(object):
    def __init__(self, nbody_loc, redshift):
        self.input_loc = nbody_loc #'/bsuhome/hwu/scratch/uchuu/Uchuu/'

        self.hubble = 0.6774 #snap_header.HubbleParam 
        self.OmegaM = 0.3089 #snap_header.Omega0
        self.boxsize = 2000 #400 #snap_header.BoxSize
        self.mpart = 6.54e+11 #snap_header.mass[1] / 5e-4 #subsmapling
        self.redshift = redshift

        if self.redshift == 0.3:
            self.snap_name = '043'
        if self.redshift == 0.1:
            self.snap_name = '047'

    def read_halos(self, Mmin=1e11, pec_vel=False, cluster_only=False):
        if cluster_only == True:
            fname = self.input_loc+f'host_halos_{self.snap_name}_M12.5.fit'
        else:
            fname = self.input_loc+f'host_halos_{self.snap_name}.fit'

        data = fitsio.read(fname)
        M200m = data['M200m']
        sel = (M200m >= Mmin)
        M200m = M200m[sel]
        sort = np.argsort(-M200m)
        self.mass = M200m[sort]

        self.hid = data['haloid'][sel][sort]
        self.xh = data['px'][sel][sort]
        self.yh = data['py'][sel][sort]
        self.zh = data['pz'][sel][sort]

        if pec_vel == True: ## takes too long to read in velocity
            self.vx = data['vx'][sel][sort]
            self.vy = data['vy'][sel][sort]
            self.vz = data['vz'][sel][sort]

    def read_particle_positions(self):
        fname = self.input_loc+f'particles_{self.snap_name}_0.05percent.h5'
        f = h5py.File(fname, 'r')
        data = f['part']
        self.xp = data['x']
        self.yp = data['y']
        self.zp = data['z']
        return self.xp, self.yp, self.zp

    def read_particle_velocities(self):
        # note: it's impossible to load both particles and velocities at the same time
        fname = self.input_loc+f'particles_{self.snap_name}_0.05percent_velocities.h5'
        f = h5py.File(fname, 'r')
        data = f['part']
        self.vxp = data['vx']
        self.vyp = data['vy']
        self.vzp = data['vz']
        return self.vxp, self.vyp, self.vzp

    def read_particles_layer(self, pz_min, pz_max):
        fname = self.input_loc+f'layers_{self.snap_name}/particles_pz_{pz_min}_{pz_max}_0.05percent.h5'
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
    import timeit
    start = timeit.default_timer()
    rmu = ReadUchuu(nbody_loc='/bsuhome/hwu/scratch/uchuu/Uchuu/', redshift=0.3)
    # rmu.read_particle_positions()
    # rmu.read_particle_velocities()
    rmu.read_particles_layer(20.0, 40.0)
    print(len(rmu.xp), len(rmu.vxp))
    #rmu.read_halos()
    #print('max', np.max(rmu.vx))
    #print('mean, std', np.mean(rmu.vx), np.std(rmu.vx))
    stop = timeit.default_timer()
    print(f'reading particles took {(stop - start):.2g} seconds')
    #reading particles took 4.2e+02 seconds
