#!/usr/bin/env python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

Pmem_2D_file = 'data/2d_RovRLambda_pmem_map_L15_z0p5.txt'
Pmem_2d_data = np.genfromtxt(Pmem_2D_file)

delz_bins = np.linspace(-0.15, 0.15, 16)
R_bins    = np.linspace(0.0, 1.0, 8)


delz_vals = (delz_bins[1:] + delz_bins[:-1])/2.0
R_vals    = (R_bins[1:] + R_bins[:-1])/2.0

pmem_interp = RegularGridInterpolator((delz_vals, R_vals), Pmem_2d_data)

Rmin = min(R_vals)
Rmax = max(R_vals)

zmin = min(delz_vals)
zmax = max(delz_vals)

dz_min = zmin
def pmem_weights(dz, R, dz_max=zmax):
    pmem_list = np.zeros(len(dz))
    
    # need to restrict to interpolated region
    sel = (dz >= dz_min)&(dz <= dz_max)&(R >= Rmin)&(R <= Rmax)
    pmem_list[sel] = pmem_interp((dz[sel], R[sel]))
    
    # very small R => set pmem to the smallest R value
    sel2 = (dz >= dz_min)&(dz <= dz_max)&(R < Rmin)
    pmem_list[sel2] = pmem_interp((dz[sel2], R[sel2]*0 + Rmin))

    return pmem_list