#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys
from astropy.io import fits
from hod.utils.get_flamingo_info import get_flamingo_cosmo, get_snap_name

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

f = h5py.File(fname,'r')

if halo_finder == 'HBT':
    M200m = f['SO/200_mean/TotalMass'][:] * 1e10 * hubble
if halo_finder == 'VR':
    M200m = f['SO/200_mean/TotalMass'][:] * hubble

Mmin = 1e11
sel2 = (M200m >= Mmin)

if halo_finder == 'HBT':
    hid_host = f['InputHalos/HaloCatalogueIndex'][:][sel2]
    pos = f['InputHalos/HaloCentre']
    M200m = f['SO/200_mean/TotalMass'][:][sel2] * 1e10 * hubble

if halo_finder == 'VR':
    hid_host = f['VR/ID'][:][sel2]
    pos = f['VR/CentreOfPotential']
    M200m = f['SO/200_mean/TotalMass'][:][sel2] * hubble


xh = pos[:,0][sel2] * hubble
yh = pos[:,1][sel2] * hubble
zh = pos[:,2][sel2] * hubble
R200m = f['SO/200_mean/SORadius'][:][sel2] * hubble 

if halo_finder == 'HBT':
    vel = f['BoundSubhalo/CentreOfMassVelocity']
if halo_finder == 'VR':
    vel = f['BoundSubhaloProperties/CentreOfMassVelocity']

vx = vel[:,0][:][sel2]
vy = vel[:,1][:][sel2]
vz = vel[:,2][:][sel2]
print('done selecting', len(vx))


############ fits ################
cols=[
    fits.Column(name='hid_host', unit='', format='K', array=hid_host),
    fits.Column(name='M200m', unit='Msun/h', format='E', array=M200m),
    fits.Column(name='R200m', unit='cMpc/h', format='E', array=R200m),
    #fits.Column(name='Mvir', unit='Msun/h', format='E', array=Mvir),
    #fits.Column(name='Rvir', unit='cMpc/h', format='E', array=Rvir),
    #fits.Column(name='Rs_Klypin', unit='', format='E', array=rs),
    fits.Column(name='px', unit='cMpc/h', format='D', array=xh),
    fits.Column(name='py', unit='cMpc/h', format='D', array=yh),
    fits.Column(name='pz', unit='cMpc/h', format='D', array=zh),
    fits.Column(name='vx', unit='', format='D', array=vx),
    fits.Column(name='vy', unit='', format='D', array=vy),
    fits.Column(name='vz', unit='', format='D', array=vz),
]
coldefs = fits.ColDefs(cols)
tbhdu = fits.BinTableHDU.from_columns(coldefs)
tbhdu.writeto(output_loc+f'host_halos_{snap_name}_{halo_finder}_M200m_{Mmin:.0e}.fit', overwrite=True)


#### read it back ####
# import fitsio
# data, header = fitsio.read(output_loc+f'host_halos_{snap_name}.fit', header=True)
# print(header)

############ hdf5 ################
'''
Nhalo = len(xh)

halo_output_dtype = np.dtype([
    ("hid_host", np.uint64, 1),
    ("Mvir", np.float32, 1),
    ("M200m", np.float32, 1),
    ("x", np.float32, 1), 
    ("y", np.float32, 1), 
    ("z", np.float32, 1),
    ("vx", np.float32, 1), 
    ("vy", np.float32, 1), 
    ("vz", np.float32, 1),
    ("Rvir", np.float32, 1),
    ])

halos_output = np.empty((Nhalo,), dtype=halo_output_dtype)

halos_output['hid_host']  = hid_host.astype(np.uint64)
halos_output['Mvir'] = Mvir.astype(np.float32)
halos_output['M200m'] = M200m.astype(np.float32)
halos_output['Rvir'] = Rvir.astype(np.float32)
halos_output['x']  = xh.astype(np.float32)
halos_output['y']  = yh.astype(np.float32)
halos_output['z']  = zh.astype(np.float32)
halos_output['vx'] = vx.astype(np.float32)
halos_output['vy'] = vy.astype(np.float32)
halos_output['vz'] = vz.astype(np.float32)

file_name = output_loc + f'host_halos_{snap_name}.h5' #<name of output h5 file>
dataset_name = 'halos'
h5f = h5py.File(file_name, 'w')
h5f.create_dataset("halos", (Nhalo,), dtype = halo_output_dtype, data = halos_output, chunks = True, compression = "gzip")
h5f.flush()
'''

