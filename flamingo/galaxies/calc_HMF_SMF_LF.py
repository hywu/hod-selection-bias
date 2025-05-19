#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import h5py
import fitsio
import os, sys
from get_flamingo_info import get_flamingo_cosmo, get_snap_name

def calc_HMF(sim_name, halofinder='HBT', mdef='vir'):
    loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
    redshift = 0.3
    snap_name = get_snap_name(sim_name, redshift)
    cosmo = get_flamingo_cosmo(sim_name)
    h = cosmo['h']

    fname = loc + f'SOAP-{halofinder}/halo_properties_{snap_name}.hdf5'
    f = h5py.File(fname,'r')

    if mdef == 'vir':
        Mhalo = f['SO/BN98/TotalMass'][:] * h # to make it Msun/h
    if mdef == '200m':
        Mhalo = f['SO/200_mean/TotalMass'][:]* h # to make it Msun/h

    if halofinder == 'HBT':
        Mhalo = Mhalo * 1e10

    Mhalo = Mhalo[Mhalo > 7e10]
    bins = np.linspace(np.log(2e10), np.log(2e15), 50)
    hist_data = np.histogram(np.log(Mhalo), bins=bins)
    n = hist_data[0]
    bins = hist_data[1]
    n = np.append(0, n)
    #n = np.append(n, 0)
    #bins = np.append(bins, bins[-1])
    V = (1000*h)**3 
    binsize = bins[1] - bins[0]
    sel = (n > -1)
    x_data = bins[sel]
    y_data = n[sel]/V/binsize
    
    output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/stats/'
    if os.path.isdir(output_loc) == False: os.makedirs(output_loc)
    data = np.array([np.exp(x_data), y_data]).transpose()
    np.savetxt(output_loc+f'HMF_{halofinder}_{mdef}.dat', data, fmt='%-12g', header=' Mhalo [Msun/h], dndlnM [h^3 Mpc^{-3}]')


def calc_SMF(sim_name, halofinder='HBT'):
    loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
    redshift = 0.3
    snap_name = get_snap_name(sim_name, redshift)
    cosmo = get_flamingo_cosmo(sim_name)
    h = cosmo['h']
    
    fname = loc + f'SOAP-{halofinder}/halo_properties_{snap_name}.hdf5'
    f = h5py.File(fname,'r')
    if halofinder == 'VR':
        halos = f['BoundSubhaloProperties']
    if halofinder == 'HBT':
        halos = f['BoundSubhalo']

    #### Stellar Mass Function ####
    Mstar = halos['StellarMass'][:]
    if halofinder == 'HBT':
        Mstar = Mstar * 1e10
    Mstar = Mstar[Mstar > 0]
    bins = np.linspace(np.log10(1e9), np.log10(3e13), 20)
    hist_data = np.histogram(np.log10(Mstar), bins=bins)
    gal_density = len(Mstar)/(1000*h)**3
    print('gal_density', gal_density, '(Mpc/h)^3')

    n = hist_data[0]
    bins = hist_data[1]
    n = np.append(0, n)
    #n = np.append(n, 0)
    #bins = np.append(bins, bins[-1])
    vol = (1000 *h)**3 # Mpc/h
    binsize = bins[1]-bins[0]
    x_data = bins
    y_data = n/vol/binsize

    output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/stats/'
    if os.path.isdir(output_loc) == False: os.makedirs(output_loc)
    data = np.array([10**x_data, y_data]).transpose()
    np.savetxt(output_loc+f'SMF_{halofinder}.dat', data, fmt='%-12g', header=' Mstar[Msun], dndlog10M [h^3 Mpc^{-3}]')


def calc_LF(sim_name, halofinder='HBT', band='i'):
    loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
    redshift = 0.3

    snap_name = get_snap_name(sim_name, redshift)
    cosmo = get_flamingo_cosmo(sim_name)
    h = cosmo['h']
    
    fname = loc + f'SOAP-{halofinder}/halo_properties_{snap_name}.hdf5'
    f = h5py.File(fname,'r')
    
    if halofinder == 'VR':
        halos = f['BoundSubhaloProperties']
    if halofinder == 'HBT':
        halos = f['BoundSubhalo']

    L = halos['StellarLuminosity'] # Total stellar luminosity in the 9 GAMA bands.
    if band=='u': col = 0
    if band=='g': col = 1 
    if band=='r': col = 2 
    if band=='i': col = 3 
    if band=='z': col = 4 

    L_band = L[:,col] 
    L_band = L_band[L_band > 0]

    bins = np.linspace(np.log10(1e8), np.log10(1e12), 30)
    hist_data = np.histogram(np.log10(L_band), bins=bins)

    n = hist_data[0]
    bins = hist_data[1]
    n = np.append(0, n)
    #n = np.append(n, 0)
    #bins = np.append(bins, bins[-1])
    vol = (1000 * h)**3 # (Mpc/h)^-3!
    binsize = bins[1]-bins[0]
    x_data = bins
    y_data = n/vol/binsize
    #plt.step(x_data, y_data)
    #plt.plot(x_data, y_data)
    #plt.yscale('log')
    
    output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/stats/'
    if os.path.isdir(output_loc) == False: os.makedirs(output_loc)
    data = np.array([x_data, y_data]).transpose()
    np.savetxt(output_loc+f'LF_{halofinder}_{band}.dat', data, fmt='%-12g', header='log10 L, dndlog10L [(Mpc/h)^-3]')

    ngal = len(L_band)/vol
    np.savetxt(output_loc+f'ngal_{halofinder}_{band}.dat', np.array([ngal]), fmt='%-12g', header='ngal [(Mpc/h)^-3]')


if __name__ == "__main__":
    sim_name = sys.argv[1]
    # sim_name1 = 'L1000N0900/HYDRO_FIDUCIAL'
    # sim_name2 = 'L1000N1800/HYDRO_FIDUCIAL'
    # sim_name3 = 'L1000N3600/HYDRO_FIDUCIAL'
    # sim_name_list = [sim_name1, sim_name2, sim_name3]
    # for sim_name in sim_name_list:
    #calc_HMF(sim_name)
    #calc_SMF(sim_name)


    for halofinder in ['VR', 'HBT']:
        calc_SMF(sim_name, halofinder)
        for mdef in ['vir', '200m']:
            calc_HMF(sim_name, halofinder, mdef)
        for band in ['g','r','i','z']:
            calc_LF(sim_name, halofinder, band)