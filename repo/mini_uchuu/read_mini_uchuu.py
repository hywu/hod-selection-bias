#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py, fitsio
import os, sys
sys.path.append('../utils')
from readGadgetSnapshot import readGadgetSnapshot

class ReadMiniUchuu(object):
    def __init__(self):
        self.input_loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
        snap_header = readGadgetSnapshot(self.input_loc+f'snapdir_043/MiniUchuu_043.gad.0')

        self.hubble = snap_header.HubbleParam 
        self.OmegaM = snap_header.Omega0
        # self.redshift = snap_header.redshift
        # self.scale_factor = 1/(1+redshift)
        self.boxsize = snap_header.BoxSize
        self.mpart = snap_header.mass[1] * 1e10 * 1000

    def read_halos(self, Mmin=1e11):
        data = h5py.File(self.input_loc+f'MiniUchuu_halolist_z0p30.h5', 'r')
        hid = np.array(data['id'])
        pid = np.array(data['pid'])
        sel1 = (pid == -1)
        M200m = np.array(data['M200b'])[sel1]
        sel2 = (M200m >= Mmin)
        self.M200m = M200m[sel2]
        self.gid = hid[sel1][sel2]
        self.xh = np.array(data['x'])[sel1][sel2]
        self.yh = np.array(data['y'])[sel1][sel2]
        self.zh = np.array(data['z'])[sel1][sel2]
        return self.xh, self.yh, self.zh, self.gid, self.M200m

    def read_particles(self):
        data = fitsio.read(self.input_loc+'particles_043_0.1percent.fit')
        xp = data['x']
        yp = data['y']
        zp = data['z']
        return xp, yp, zp
