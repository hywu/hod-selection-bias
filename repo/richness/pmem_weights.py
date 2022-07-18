#!/usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

Pmem_2D_file = 'data/2d_RovRLambda_pmem_map_L15_z0p5.txt'
Pmem_2d_data = np.genfromtxt(Pmem_2D_file)

delz_bins = np.linspace(-0.15, 0.15, 16) #Of course we can change this, but it's what I have computed so far
R_bins    = np.linspace(0.0, 1.0, 8)


delz_vals = (delz_bins[1:] + delz_bins[:-1])/2.0
R_vals    = (R_bins[1:] + R_bins[:-1])/2.0

pmem_interp = RegularGridInterpolator((delz_vals, R_vals), Pmem_2d_data)

def pmem_weights(dz, R):
    pmem_list = np.zeros(len(dz))
    
    # need to restrict to interpolated region
    sel = (dz > min(delz_vals))&(dz < max(delz_vals))&(R > min(R_vals))&(R < max(R_vals))  
    pmem_list[sel] = pmem_interp((dz[sel], R[sel]))
    
    # very small R => set pmem = 1
    sel2 = (dz > min(delz_vals))&(dz < max(delz_vals))&(R < min(R_vals))
    pmem_list[sel2] = 1
    
    return pmem_list