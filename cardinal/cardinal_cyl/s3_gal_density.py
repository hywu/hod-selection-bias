#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import fitsio
import h5py
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.286)

# use gold mask to get area
cat_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
fname = cat_loc + 'Cardinal-3_v2.0_Y6a_gold.h5'
f = h5py.File(fname, 'r')
hpix = f['masks/gold/hpix'][:]
npix = len(hpix)
import healpy
nside = 4096
npix_tot = healpy.nside2npix(nside)
fsky = npix/npix_tot
print('gold area [sq deg] = ',  fsky * 41253.)

# get galaxy catalog
chisq_cut = int(sys.argv[1])#100
output_loc = '/projects/hywu/cluster_sims/cluster_finding/data/cardinal_cyl/'
output_loc += f'model_chisq{chisq_cut}/'
fname = output_loc + 'gals.fit'
data, header = fitsio.read(fname, header=True)
print(header)
chi = data['chi']


def get_gal_density(zmin, zmax):

    # volume in this bin
    vol_allsky = cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value
    vol = vol_allsky * fsky
    print('gold volume %g [hiGpc^3]'%(vol * 1e-9))

    # count galaxies
    chi_min = cosmo.comoving_distance(zmin).value
    chi_max = cosmo.comoving_distance(zmax).value
    sel = (chi > chi_min)&(chi < chi_max)
    den = len(chi[sel])/vol
    return den


z_list = np.arange(0,1,0.01)
nz = len(z_list) - 1
outfile = open(output_loc+'gal_density.dat', 'w')
outfile.write('# z, ngal [hiMpc^{-3}]\n')
for iz in range(nz):
    zmin = z_list[iz]
    zmax = z_list[iz+1]
    zmid = 0.5*(zmin + zmax)
    ngal = get_gal_density(zmin, zmax)
    print(zmid, ngal)
    outfile.write('%12g %12g\n'%(zmid, ngal))
outfile.close()