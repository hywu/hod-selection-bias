#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys

for sim_name in ['L1000N3600/HYDRO_FIDUCIAL','L1000N3600/DMO_FIDUCIAL']:
    for halo_finder in ['HBT', 'VR']: # 
        os.system(f'./get_host_halos.py {sim_name} {halo_finder} &')

