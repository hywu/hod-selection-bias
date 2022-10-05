#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py, fitsio
import os, sys
sys.path.append('../utils')
from readGadgetSnapshot import readGadgetSnapshot

class ReadMiniUchuu(object):
    def __init__(self, nbody_loc):
        self.input_loc = nbody_loc #'/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
        snap_header = readGadgetSnapshot(self.input_loc+f'snapdir_043/MiniUchuu_043.gad.0')

        self.hubble = snap_header.HubbleParam 
        self.OmegaM = snap_header.Omega0
        self.boxsize = snap_header.BoxSize
        self.mpart = snap_header.mass[1] * 1e10 * 1000

    def read_halos(self, Mmin=1e11):
        data = fitsio.read(self.input_loc+'host_halos_043.fit')
        M200m = data['M200m']
        sel = (M200m >= Mmin)
        M200m = M200m[sel]
        hid = data['haloid'][sel]
        xh = data['px'][sel]
        yh = data['py'][sel]
        zh = data['pz'][sel]

        sort = np.argsort(-M200m)
        self.mass = M200m[sort]
        self.hid = hid[sort]
        self.xh = xh[sort]
        self.yh = yh[sort]
        self.zh = zh[sort]
        return self.hid, self.mass, self.xh, self.yh, self.zh

    def read_particles(self):
        data = fitsio.read(self.input_loc+'particles_043_0.1percent.fit')
        xp = data['x']
        yp = data['y']
        zp = data['z']
        return xp, yp, zp

if __name__ == '__main__':
    rmu = ReadMiniUchuu()
    x1, y1, z1, hid1, M1 = rmu.read_halos_new()
    x2, y2, z2, hid2, M2 = rmu.read_halos_old()
    print('test x',max(abs(x1-x2)))
    print('test hid',max(abs(hid1-hid2)))
    print('test M %e'%max(abs(M1-M2)))

