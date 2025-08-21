#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import sys
import h5py
from hod.utils.get_flamingo_info import get_flamingo_cosmo, get_snap_name

#import swiftsimio as sw
redshift = 0.4 #0.4 #0.3

sim_name = 'L1000N3600/HYDRO_FIDUCIAL'
input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/z{redshift}/'
snap_name = get_snap_name(sim_name, redshift)


# original data: 480 files, 8.4 G per file
# downsample 1e-4 => 400 MB total, mpart: 1e12 (too large)
# downsample 1e-3 => 4 GB total, mpart: 1e11

x = []
y = []
z = []

import glob
files = glob.glob(input_loc+f'snapshots/flamingo_{snap_name}/flamingo_{snap_name}.*.hdf5')
nfiles = len(files)


for ifile in range(nfiles):
    file_name = output_loc + f'temp/particles_{snap_name}_0.1percent.{ifile}.h5'
    data = h5py.File(file_name, 'r')
    data = data['part']
    x.extend(data['x'])
    y.extend(data['y'])
    z.extend(data['z'])
    print(ifile, np.shape(x))


x = np.array(x)
y = np.array(y)
z = np.array(z)

npart = len(x)

part_output_dtype = np.dtype([
    ("x", (np.float32,1)), 
    ("y", (np.float32,1)), 
    ("z", (np.float32,1))
    ])

part_output = np.empty((npart,), dtype=part_output_dtype)

part_output['x']    = x.astype(np.float32)
part_output['y']    = y.astype(np.float32)
part_output['z']    = z.astype(np.float32)

file_name = output_loc + f'particles_{snap_name}_0.1percent.h5' #<name of output h5 file>
os.system(f'rm -rf {file_name}')
dataset_name = 'particles'
h5f = h5py.File(file_name, 'w')
h5f.create_dataset("part", (npart,), dtype = part_output_dtype, data = part_output, chunks = True, compression = "gzip")
h5f.flush()
