#!/usr/bin/env python
import numpy as np

def periodic_boundary_condition(x, y, z, boxsize, x_padding, y_padding, z_padding):
    # apply periodic boundary condition & cut at the edge
    x_pbc = []
    y_pbc = []
    z_pbc = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                x_temp = x + i * boxsize
                y_temp = y + j * boxsize
                z_temp = z + k * boxsize
                selx = (x_temp > - x_padding)&(x_temp < boxsize + x_padding)
                sely = (y_temp > - y_padding)&(y_temp < boxsize + y_padding)
                selz = (z_temp > - z_padding)&(z_temp < boxsize + z_padding)
                sel = selx & sely & selz
                x_pbc.extend(x_temp[sel])
                y_pbc.extend(y_temp[sel])
                z_pbc.extend(z_temp[sel])
                
    return np.array(x_pbc), np.array(y_pbc), np.array(z_pbc)


def periodic_boundary_condition_halos(x, y, z, boxsize, x_padding, y_padding, z_padding, hid, mass):
    x_pbc = []
    y_pbc = []
    z_pbc = []
    hid_pbc = []
    mass_pbc = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                x_temp = x + i * boxsize
                y_temp = y + j * boxsize
                z_temp = z + k * boxsize
                selx = (x_temp > - x_padding)&(x_temp < boxsize + x_padding)
                sely = (y_temp > - y_padding)&(y_temp < boxsize + y_padding)
                selz = (z_temp > - z_padding)&(z_temp < boxsize + z_padding)
                sel = selx & sely & selz
                x_pbc.extend(x_temp[sel])
                y_pbc.extend(y_temp[sel])
                z_pbc.extend(z_temp[sel])
                hid_pbc.extend(hid[sel])
                mass_pbc.extend(mass[sel])
                
    return np.array(x_pbc), np.array(y_pbc), np.array(z_pbc), np.array(hid_pbc), np.array(mass_pbc)


