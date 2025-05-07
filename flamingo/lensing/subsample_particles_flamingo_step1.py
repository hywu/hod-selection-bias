#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import sys
import h5py
from hod.utils.get_flamingo_info import get_flamingo_cosmo, get_snap_name
import swiftsimio as sw

# step 1: downsample, save temp files
# step 2: merge all files

redshift = 0.5 #0.4 #0.3

sim_name = 'L1000N3600/HYDRO_FIDUCIAL'
input_loc = f'/cosma8/data/dp004/flamingo/Runs/{sim_name}/'
output_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/z{redshift}/'
snap_name = get_snap_name(sim_name, redshift)

# original data: 480 files, 8.4 GB per file
# downsample 1e-4 => 400 MB total, mpart: 1e12 (too large)
# downsample 1e-3 => 4 GB total, mpart: 1e11

import glob
files = glob.glob(input_loc+f'snapshots/flamingo_{snap_name}/flamingo_{snap_name}.*.hdf5')
nfiles = len(files)

for ifile in range(nfiles):
    outputfile_name = output_loc + f'temp/particles_{snap_name}_0.1percent.{ifile}.h5' #<name of output h5 file>
    if os.path.exists(outputfile_name):
        pass
    else:
        data = sw.load(input_loc+f'snapshots/flamingo_{snap_name}/flamingo_{snap_name}.{ifile}.hdf5') 
        x = data.dark_matter.coordinates.value[:,0][::1000]
        y = data.dark_matter.coordinates.value[:,1][::1000]
        z = data.dark_matter.coordinates.value[:,2][::1000]
        print('doing ', ifile)

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

        #os.system(f'rm -rf {outputfile_name}')
        dataset_name = 'particles'
        h5f = h5py.File(outputfile_name, 'w')
        h5f.create_dataset("part", (npart,), dtype = part_output_dtype, data = part_output, chunks = True, compression = "gzip")
        h5f.flush()
