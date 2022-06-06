#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

import os, h5py, glob

depth = 30#15 #60 #30

radius = 1.0

for phase in [0]:#range(1,20):

    loc_out = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/' # output location

    loc_in = f'/bsuhome/hwu/scratch/Projection_Effects/Catalogs/fiducial-{phase}/z0p3/' # data location
    os.chdir(loc_in)
    #hod_list = glob.glob('memHOD*z0p3.hdf5')
    hod_list = [f'memHOD_11.2_12.4_0.65_1.0_0.2_0.0_{phase}_z0p3.hdf5'] # fid

    os.chdir('/bsuhome/hwu/work/hod-selection-bias/repo/richness') # current location

    for hod in hod_list:
        plt.figure()
        hod = hod[:-5]
        loc = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/'

        #cl_fname = loc + 'memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3.richness_d30.hdf5'
        cl_fname = loc + hod + '/'+ hod + f'.richness_d{depth}_r{radius}.hdf5'

        f = h5py.File(cl_fname,'r')
        halos = f['halos']
        print(halos.dtype)
        mass = halos['mass']
        lam = halos['lambda']

        sel = (lam > 0)
        plt.scatter(mass[sel], lam[sel], s=1, alpha=0.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.axvline(10**12.5, c='gray')
        plt.xlabel('mass')
        plt.ylabel('richness')
        plt.savefig(f'../../plots/richness/sanity/fiducial-{phase}_{hod}.png')
#plt.show()
