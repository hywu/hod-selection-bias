#!/usr/bin/env python
import numpy as np


x_halo, y_halo, z_halo = np.loadtxt('halo.dat')
x_gal, y_gal, z_gal = np.loadtxt('gal.dat', unpack=True)
# replace this by a generous radius

r = (x_gal - x_halo)**2 + (y_gal - y_halo)**2 

def rlambda(r, rlam_ini):
    rlam = rlam_ini
    for iteration in range(10):
        sel = (r < rlam)
        ngal = len(r[sel])
        rlam_old = rlam
        rlam = (ngal/100.)**0.2
        #print(rlam, rlam_old)
        if abs(rlam - rlam_old) < 1e-5:
            break
    return rlam, len(r < rlam)

print(rlambda(r, 2))