#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import fitsio
import os
import sys
#sys.path.append('../utils')
#from readGadgetSnapshot import readGadgetSnapshot

class ReadAbacusSummit(object):
    def __init__(self, nbody_loc):
        self.input_loc = nbody_loc
        #self.input_loc = '/bsuhome/hwu/scratch/abacus_summit/'


        # TODO: how to read the header?
        #snap_header = self.input_loc+'header'

        self.hubble = 0.6736
        self.OmegaM = 0.31519
        self.boxsize = 2e3
        self.mpart = 2.109e+09 / 0.003

    def read_halos(self, Mmin=1e11):
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
        halo_fname = self.input_loc+'halos_1e+11.fit'
        #print(halo_fname)
        data = fitsio.read(halo_fname)
        mass = data['mass']
        sel = (mass > Mmin)
        mass = mass[sel]
        hid = data['haloid'][sel]
        xh = data['px'][sel]
        yh = data['py'][sel]
        zh = data['pz'][sel]

        sort = np.argsort(-mass)
        self.mass = mass[sort]
        self.hid = hid[sort]
        self.xh = xh[sort]
        self.yh = yh[sort]
        self.zh = zh[sort]
        return self.hid, self.mass, self.xh, self.yh, self.zh

    def read_particles(self):
        part_fname = self.input_loc + 'subsample_particles_A_base_c000_ph000_z0p300.h5'
        f = h5py.File(part_fname, 'r')
        particles = f['particles']
        xp = particles['x']
        yp = particles['y']
        zp = particles['z']
        return xp, yp, zp

if __name__ == '__main__':
    ras = ReadAbacusSummit()
    #x, y, z, hid, m = ras.read_halos(Mmin=1e13)
    x, y, z = ras.read_particles()


