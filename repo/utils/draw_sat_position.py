#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import default_rng
rng = default_rng()
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import concentration
from colossus.halo import profile_nfw

ombh2 = 0.02222
omcdmh2 = 0.1199
omh2 = 0.14212
w0 = -1.0
wa = 0
ns = 0.9652
sigma_8 = 0.830
H0 = 67.26
h = H0/100
Om0 = omh2/(h**2)
Ob0 = ombh2/(h**2)

params = {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma_8, 'ns': ns}
cosmo = cosmology.setCosmology('AbacusCosmo', params)
#print(cosmo)

redshift = 0.3 

M200m_interp = 10**np.linspace(11,15.5,10)
c_interp = concentration.concentration(M200m_interp, z=redshift, mdef='200m', model='bhattacharya13')
c_lnM_interp = interp1d(np.log(M200m_interp), c_interp)



def draw_sat_position(redshift, M200m, Nsat):
    a = 1./ (1. + redshift)
    rhocrit = 2.775e11 # h-unit
    Om0 = 0.314
    R200m_cMpc = (3 * M200m / (4 * np.pi * rhocrit * Om0 * 200.))**(1./3.) # cMpc/h (chimp)
    R200m = R200m_cMpc * 1e3 * a # pkpc
    # print(max(R200m))
    #R200m = mass_so.M_to_R(M = M200m, z=redshift, mdef='200m')
    #c = concentration.concentration(M200m, z=redshift, mdef='200m', model='bhattacharya13')
    c = c_lnM_interp(np.log(M200m))
    # p_nfw = profile_nfw.NFWProfile(M = M200m, c = c, z = redshift, mdef = '200m')
    # r_nfw = 10**np.linspace(-4, np.log10(R200m), 1000)  # pkpc
    # cdf = p_nfw.cumulativePdf(r_nfw, Rmax=R200m, z=redshift, mdef='200m')
    r_nfw = 10**np.linspace(-4, np.log10(R200m), 1000)  # pkpc
    rs = R200m / c
    cdf = 4 * np.pi * rs**3 * (np.log((r_nfw + rs)/rs) - r_nfw/(r_nfw + rs) ) / M200m
    cdf /= cdf[-1]

    r_cdf_interp = interp1d(cdf, r_nfw)
    rand_vals = rng.random(Nsat)
    rsat_pkpc = r_cdf_interp(rand_vals)
    
    rsat_cMpc = rsat_pkpc / 1e3 / a

    u = rng.random(Nsat)
    v = rng.random(Nsat)
    theta = 2 * np.pi * u 
    phi = np.arccos(2 * v - 1)
    px = rsat_cMpc * np.cos(theta) * np.sin(phi)
    py = rsat_cMpc * np.sin(theta) * np.sin(phi)
    pz = rsat_cMpc * np.cos(phi)

    if max(rsat_cMpc) > R200m_cMpc: print('outside R200m')
    return px, py, pz

def calc_sat_density(rsat, label=None):
    rmin = min(rsat)
    rmax = max(rsat)
    den = []
    rmean = []
    nr = 20
    r_bins = 10.**np.linspace(np.log10(rmin), np.log10(rmax), nr+1)
    for ir in range(nr):
        rsel = rsat[(rsat > r_bins[ir])&(rsat <= r_bins[ir+1])]
        rmid = np.mean(rsel)
        rmean.append(rmid)
        n = len(rsel)
        vol = 4. * np.pi/3. * (r_bins[ir+1]**3 - r_bins[ir]**3)
        den.append(n / vol)

    rmean = np.array(rmean)
    den = np.array(den) / (1.*len(rsat))
    return rmean, den



# z = 0.3
# M200m = 1e14
# Nsat = 3

# x, y, z = draw_sat_position(z, M200m, Nsat)
# print(x, y, z)