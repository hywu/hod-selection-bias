#!/usr/bin/env python
import numpy as np
#import matplotlib.pyplot as plt
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
        self.mpart = 3.270422e+11 #snap_header.mass[1] * 1e10 * 1000 ## TODO! check
        self.redshift = redshift

        if self.redshift == 0.3:
            self.snap_name = '043'
        if self.redshift == 0.1:
            self.snap_name = '047'

    def read_halos(self, Mmin=1e11, pec_vel=False, cluster_only=False):
        if cluster_only == True:
            fname = self.input_loc+f'host_halos_{self.snap_name}_M12.5.fit'
        else:
            self.input_loc+f'host_halos_{self.snap_name}.fit'

        data = fitsio.read()
        M200m = data['M200m']
        sel = (M200m >= Mmin)
        M200m = M200m[sel]
        # hid = data['haloid'][sel]
        # xh = data['px'][sel]
        # yh = data['py'][sel]
        # zh = data['pz'][sel]

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
        # else:
        #     nh = len(self.xh)
        #     self.vx = np.zeros(nh)
        #     self.vy = np.zeros(nh)
        #     self.vz = np.zeros(nh)
        #return self.hid, self.mass, self.xh, self.yh, self.zh
    
    def read_particles(self):
        fname = self.input_loc+f'particles_{self.snap_name}_0.05percent.h5'
        f = h5py.File(fname, 'r')
        data = f['part']
        xp = data['x']
        yp = data['y']
        zp = data['z']
        return xp, yp, zp
    
if __name__ == '__main__':
    import timeit
    start = timeit.default_timer()
    rmu = ReadUchuu(nbody_loc='/bsuhome/hwu/scratch/uchuu/Uchuu/', redshift=0.1)
    rmu.read_particles()
    #rmu.read_halos(pec_vel=True) #took 1.6e+02 seconds
    #print('max', np.max(rmu.vx))
    #print('mean, std', np.mean(rmu.vx), np.std(rmu.vx))
    
    # x1, y1, z1, hid1, M1 = rmu.read_halos_new()
    # x2, y2, z2, hid2, M2 = rmu.read_halos_old()
    # print('test x',max(abs(x1-x2)))
    # print('test hid',max(abs(hid1-hid2)))
    # print('test M %e'%max(abs(M1-M2)))

    stop = timeit.default_timer()
    print(f'took {(stop - start):.2g} seconds')