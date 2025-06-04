#!/usr/bin/env python
# %load_ext autoreload
# %autoreload 2
# import numpy as np
# import matplotlib.pyplot as plt
# plt.style.use('MNRAS')
#from scipy import linalg
#from scipy.interpolate import interp1d
import h5py
import fitsio

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.286) # Mpc/h units

card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
# zmin = 0.2
# zmax = 0.35


# Gold
# use gold mask to get area
card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
fname = card_loc + 'Cardinal-3_v2.0_Y6a_gold.h5'
f = h5py.File(fname, 'r')
hpix = f['masks/gold/hpix'][:]
npix = len(hpix)
import healpy
nside = 4096
npix_tot = healpy.nside2npix(nside)
fsky = npix/npix_tot
print('gold area [sq deg] = ',  fsky * 41253.)

def volume_gold(zmin, zmax):
    vol_allsky = cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value
    vol = vol_allsky * fsky
    #print('gold volume %g [hiGpc^3]'%(vol * 1e-9))
    return vol



# Redmapper
fname = card_loc + 'redmapper_v4_v8_v51_y6_v7/run//Cardinal-3Y6a_v2.0_run_run_redmapper_v0.8.1_weighted_randoms_z010-095_lgt020_vl02_area.fit'
data, header = fitsio.read(fname, header=True)

def volume_redmapper(zmin, zmax):
    z = data['z']
    area = data['area']
    sel = (z > zmin)&(z < zmax)
    z = z[sel]
    area = area[sel]
    vol = 0
    for iz in range(len(z)-1):
        vol_allsky = cosmo.comoving_volume(z[iz+1]).value - cosmo.comoving_volume(z[iz]).value
        fsky = 0.5*(area[iz] + area[iz+1]) / 41253.
        vol += (vol_allsky * fsky)
    print(vol)
    #print('Redmapper volume = %g (Gpc/h)^3'%(vol*1e-9))
    # redmapper volume is smaller than gold
    return vol

if __name__ == "__main__":
    print(volume_gold(0.2,0.21))
    print(volume_redmapper(0.2,0.21))
