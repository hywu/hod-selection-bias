#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# magnitude limit
fname = f'../cardinal_cyl/data_member/mag_z_limit.dat'
zmin, zmax, mag_z_limit  = np.loadtxt(fname, unpack=True)
zmid = 0.5 * (zmin + zmax)
plt.plot(zmid, mag_z_limit, label='z-band from Cardinal')#, label=color_name.replace('_','-'))
mag_z_limit_interp = interp1d(zmid, mag_z_limit)

def mag_z_lim(z):
    return mag_z_limit_interp(z)


# color best fit
#### I'll make it nicer later...
################################################
# g_r
fname = f'../cardinal_cyl/data_member/g_r_model.dat'
zmin, zmax, c_intercept, c_slope, c_scatter, c_mean = np.loadtxt(fname, unpack=True)
zmid = 0.5 * (zmin + zmax)
g_r_intercept = interp1d(zmid, c_intercept)
g_r_slope = interp1d(zmid, c_slope)
g_r_scatter = interp1d(zmid, c_scatter)

def g_r_vs_mag_redshift(mag, z):
    return g_r_intercept(z) + g_r_slope(z) * mag

def sigma_g_r_vs_redshift(z):
    return g_r_scatter(z)

################################################
# r_i
fname = f'../cardinal_cyl/data_member/r_i_model.dat'
zmin, zmax, c_intercept, c_slope, c_scatter, c_mean = np.loadtxt(fname, unpack=True)
zmid = 0.5 * (zmin + zmax)
r_i_intercept = interp1d(zmid, c_intercept)
r_i_slope = interp1d(zmid, c_slope)
r_i_scatter = interp1d(zmid, c_scatter)

def r_i_vs_mag_redshift(mag, z):
    return r_i_intercept(z) + r_i_slope(z) * mag

def sigma_r_i_vs_redshift(z):
    return r_i_scatter(z)

################################################
# i_z
fname = f'../cardinal_cyl/data_member/i_z_model.dat'
zmin, zmax, c_intercept, c_slope, c_scatter, c_mean = np.loadtxt(fname, unpack=True)
zmid = 0.5 * (zmin + zmax)
i_z_intercept = interp1d(zmid, c_intercept)
i_z_slope = interp1d(zmid, c_slope)
i_z_scatter = interp1d(zmid, c_scatter)

def i_z_vs_mag_redshift(mag, z):
    return i_z_intercept(z) + i_z_slope(z) * mag

def sigma_i_z_vs_redshift(z):
    return i_z_scatter(z)


