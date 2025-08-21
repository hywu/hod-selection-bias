#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.io import fits
import os, sys
from get_flamingo_info import get_flamingo_cosmo, get_snap_name

redshift = 0.5 #0.4 #0.3

sim_name = 'L1000N3600/HYDRO_FIDUCIAL'
input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/z{redshift}/'

if os.path.isdir(output_loc) == False: 
    os.makedirs(output_loc)
    print(output_loc)

halo_finder = 'HBT' # 'VR'

scale_factor = 1./(1.+redshift)
snap_name = get_snap_name(sim_name, redshift)

cosmo = get_flamingo_cosmo(sim_name)
hubble = cosmo['h']

if halo_finder == 'HBT':
    fname = input_loc + f'SOAP-HBT/halo_properties_{snap_name}.hdf5'
if halo_finder == 'VR':
    fname = input_loc + f'SOAP-VR/halo_properties_{snap_name}.hdf5'

#### select galaxies with a Mstar limit 
#### Performing galaxy-halo matching using R200m 
#### Save a galaxy catalog with griz bands (for futher selection)

Mstar_lim = 2e+10
output_loc = output_loc+f'/model_mstar{Mstar_lim:.0e}_{halo_finder}/'
if os.path.isdir(output_loc) == False: 
    os.makedirs(output_loc)

f = h5py.File(fname,'r')

#### read subhalo properties ####
if halo_finder == 'HBT':
    id_halofinder = f['InputHalos/HaloCatalogueIndex'][:]
    M200m_list = f['SO/200_mean/TotalMass'][:] * 1e10 * hubble
if halo_finder == 'VR':
    id_halofinder = f['VR/ID'][:]
    M200m_list = f['SO/200_mean/TotalMass'][:] * hubble

R200m_list = f['SO/200_mean/SORadius'][:] * hubble 

if halo_finder == 'HBT':
    pos = f['InputHalos/HaloCentre']
if halo_finder == 'VR':
    pos = f['VR/CentreOfPotential']

x_list = pos[:,0] * hubble
y_list = pos[:,1] * hubble
z_list = pos[:,2] * hubble
if halo_finder == 'HBT':
    vel = f['BoundSubhalo/CentreOfMassVelocity']
if halo_finder == 'VR':
    vel = f['BoundSubhaloProperties/CentreOfMassVelocity']
    
vx_list = vel[:,0]
vy_list = vel[:,1]
vz_list = vel[:,2]

#### select galaxies
if halo_finder == 'HBT':
    Mstar_list = f['BoundSubhalo/StellarMass'][:] * 1e10
    L = f['BoundSubhalo/StellarLuminosity']  #u, g, r, i, z, Y, J, H, K
if halo_finder == 'VR':
    Mstar_list = f['BoundSubhaloProperties/StellarMass'][:]
    L = f['BoundSubhaloProperties/StellarLuminosity']  #u, g, r, i, z, Y, J, H, K
    
L_g_list = L[:,1]
L_r_list = L[:,2]
L_i_list = L[:,3]
L_z_list = L[:,4]


sel_gal = (Mstar_list > Mstar_lim)&(L_g_list > 0)&(L_r_list > 0)&(L_i_list > 0)&(L_z_list > 0)

x_gal = x_list[sel_gal]
y_gal = y_list[sel_gal]
z_gal = z_list[sel_gal]
vx_gal = vx_list[sel_gal]
vy_gal = vy_list[sel_gal]
vz_gal = vz_list[sel_gal]
id_gal = id_halofinder[sel_gal]
M200m_gal = M200m_list[sel_gal] # Zero for subhalo. used later for percolation

# Absolute AB magnitude. see the SOAP manual
M_g = -2.5 * np.log10(L_g_list[sel_gal])
M_r = -2.5 * np.log10(L_r_list[sel_gal])
M_i = -2.5 * np.log10(L_i_list[sel_gal])
M_z = -2.5 * np.log10(L_z_list[sel_gal])

# is a central galaxy nor not
# structype = f['VR/StructureType'][:][sel_gal]
# iscen = np.zeros(len(structype))
# iscen[structype == 10] = 1
#iscen = f['InputHalos/IsCentral'][:]


#### select halos (for matching purposes) ####
sel_halo = M200m_list > 5e14 #2e11
x_halo = x_list[sel_halo]
y_halo = y_list[sel_halo]
z_halo = z_list[sel_halo]
id_halo = id_halofinder[sel_halo]
M200m_halo = M200m_list[sel_halo]
R200m_halo = R200m_list[sel_halo]

##### Match galaxies to halos using R200m ####
boxsize = 1000 * hubble
from scipy import spatial
rmax_tree = max(R200m_halo)+0.1
gal_position = np.dstack([x_gal, y_gal, z_gal])[0]
gal_tree = spatial.cKDTree(gal_position, boxsize=boxsize)

halo_position = np.dstack([x_halo, y_halo, z_halo])[0]
halo_tree = spatial.cKDTree(halo_position, boxsize=boxsize)

indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)

nhalo = len(M200m_halo)
Ngal = []

ntot = len(x_gal)
id_host = np.zeros(ntot)
M200m_host = np.zeros(ntot)

for i_halo in range(nhalo):
    gal_ind = indexes_tree[i_halo]
    x_cen = x_halo[i_halo]
    y_cen = y_halo[i_halo]
    z_cen = z_halo[i_halo]
    R200m = R200m_halo[i_halo]
    M200m = M200m_halo[i_halo]
    hid = id_halo[i_halo]
    indx = gal_tree.query_ball_point([x_cen, y_cen, z_cen], R200m)
    Ngal.append(len(indx))

    M200m_host[indx] = M200m
    id_host[indx] = hid


##### Save the galaxy file #####

#print('frac', len(M200m_host[M200m_host==0])/len(M200m_host))
### 10% galaxies have host mass < 2e10.  Disgard them??
#sel_gal = (M200m_host > 2e11)

cols=[
  #fits.Column(name='hid_host', format='K', array=id_host),
  #fits.Column(name='hid_sub', format='K', array=id_gal),
  fits.Column(name='mass_host', unit='M200m, Msun/h', format='E', array=M200m_host),
  #fits.Column(name='mass_sub', unit='M200m, Msun/h', format='E', array=M200m_gal), # good redundancy
  fits.Column(name='px', unit='Mpc/h', format='D', array=x_gal),
  fits.Column(name='py', unit='Mpc/h', format='D', array=y_gal),
  fits.Column(name='pz', unit='Mpc/h', format='D', array=z_gal),
  fits.Column(name='vx', format='D', array=vx_gal),
  fits.Column(name='vy', format='D', array=vy_gal),
  fits.Column(name='vz', format='D', array=vz_gal),
  #### Magnitude information ####
  fits.Column(name='M_g', unit='', format='D', array=M_g),
  fits.Column(name='M_r', unit='', format='D', array=M_r),
  fits.Column(name='M_i', unit='', format='D', array=M_i),
  fits.Column(name='M_z', unit='', format='D', array=M_z),
  #fits.Column(name='iscen', unit='', format='K',array=iscen),
]
coldefs = fits.ColDefs(cols)
tbhdu = fits.BinTableHDU.from_columns(coldefs)
fname = f'gals.fit'
tbhdu.writeto(f'{output_loc}/{fname}', overwrite=True)
