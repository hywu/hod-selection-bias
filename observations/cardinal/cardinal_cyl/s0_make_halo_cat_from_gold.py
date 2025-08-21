#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import h5py
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.286)
z_list = np.arange(0,5,0.0001)
chi_list = cosmo.comoving_distance(z_list).value
redshift_chi_interp = interp1d(chi_list, z_list)


card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
fname = card_loc + 'Cardinal-3_v2.0_Y6a_gold.h5'
f = h5py.File(fname, 'r')
data = f['catalog/gold']

rhalo = data['rhalo'][:]
mass = data['m200'][:]
sel = (rhalo == 0)&(mass > 0)

mass = mass[sel]

haloid = data['haloid'][sel]
radius = data['r200'][sel]
ra_all = data['ra'][sel]
dec_all = data['dec'][sel]
px_all = data['px'][sel]
py_all = data['py'][sel]
pz_all = data['pz'][sel]
chi_all = np.sqrt(px_all**2 + py_all**2 + pz_all**2) # there is no redshift. sigh.
z_all = redshift_chi_interp(chi_all)


# sort by mass
index = np.argsort(-mass)
mass = mass[index]

haloid = haloid[index]
radius = radius[index]
ra_all = ra_all[index]
dec_all = dec_all[index]
px_all = px_all[index]
py_all = py_all[index]
pz_all = pz_all[index]
chi_all = chi_all[index]
z_all = z_all[index]



cols=[
  fits.Column(name='haloid', format='K', array=haloid),
  fits.Column(name='mvir', unit='Msun/h', format='E', array=mass),
  fits.Column(name='rvir', unit='Msun/h', format='D', array=radius),
  fits.Column(name='ra', unit='deg, rotated', format='D', array=ra_all),
  fits.Column(name='dec', unit='deg, rotated', format='D', array=dec_all),
  fits.Column(name='px', unit='Mpc/h', format='D', array=px_all),
  fits.Column(name='py', unit='Mpc/h', format='D', array=py_all),
  fits.Column(name='pz', unit='Mpc/h', format='D', array=pz_all),
  fits.Column(name='chi', unit='Mpc/h', format='D', array=chi_all),
  fits.Column(name='z_cos', unit='redshift', format='D', array=z_all),
]
coldefs = fits.ColDefs(cols)
tbhdu = fits.BinTableHDU.from_columns(coldefs)

output_loc = '/projects/hywu/cluster_sims/cluster_finding/data/cardinal_cyl/'
fname = 'halos_from_gold.fit'
tbhdu.writeto(f'{output_loc}/{fname}', overwrite=True)


