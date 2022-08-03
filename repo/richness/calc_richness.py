#!/usr/bin/env python
import timeit
start = timeit.default_timer()

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys
from distutils import util
import config

import argparse


'TODO'
'probabilistic percolation'

# example
# ./calc_richness.py --phase 0 --run_name memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3 --use_pmem
# ./calc_richness.py --phase 0 --run_name memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3 --use_cylinder --depth 1

## required
parser = argparse.ArgumentParser()
parser.add_argument('--halos', required=True, help='File name of halo hdf5 file')
parser.add_argument('--members', required=True, help='File name of mock member galaxies hdf5 file')
parser.add_argument('--header', required=True, help='Header file name for simulation')

group = parser.add_mutually_exclusive_group() # cylinder or pmem?
group.add_argument('--use_cylinder', action='store_true', help='choose either use_cylidner or use_pmem')
group.add_argument('--use_pmem', action='store_true', help='choose either use_cylinder or use_pmem')

## optional
parser.add_argument('--depth', type=float, help='required if use_cylinder==True') # 30

parser.add_argument('--input_path', help='path to your input hdf5 files')
parser.add_argument('--output_path', help='path to your output hdf5 files')

parser.add_argument('--fix_radius', action='store_true', help='whether we use rlambda or fixed radius')
parser.add_argument('--radius', type=float, help='required if fix_radius == True')

parser.add_argument('--noperc', action='store_true', help='turn off percolation')

parser.add_argument('--ID_str', help='Unique identifying string for parallel computations')


args = parser.parse_args()

halo_file   = args.halos
memgal_file = args.members

use_cylinder = args.use_cylinder
use_pmem = args.use_pmem

## cylinder or pmem?
if args.use_cylinder == True:
    depth = args.depth
    print('using cylinder with depth', depth)

if args.use_pmem == True:
    use_pmem = True
    print('using pmem')


## rlambda or fixed radius?
if args.fix_radius == True:
    use_rlambda = False
    radius = args.radius
    print('using fixed radius ', radius, 'cMpc/h')
else:
    use_rlambda = True
    print('use rlambda')

if args.ID_str:
    ofname_base = str(args.ID_str)+f'_richness'
else:
    ofname_base = f'richness'
    
if use_cylinder == True:
    ofname_base  += f'_d{depth}'
if args.fix_radius == True:
    ofname_base += f'_r{radius}'
if args.noperc == True:
    ofname_base += '_noperc'




## input & output paths
if args.input_path:
    in_path = args.input_path
else:
    in_path = f'/bsuhome/hwu/scratch/Projection_Effects/Catalogs/fiducial-{phase}/z0p3/'

if args.output_path:
    out_path = args.output_path
else:
    out_path = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/{run_name}'

if args.noperc == True:
    no_perc = args.noperc
    perc = False
else:
    perc = True

if os.path.isdir(out_path)==False:
    os.makedirs(out_path)

print('input path', in_path)
print('output path', out_path)


#### things can be moved to yml files ####

#boxsize = 1100
#redshift = 0.3
#scale_fac = 1/(1+redshift)
#OmegaM = 0.314 
#hubble = 0.67

cf = config.AbacusConfigFile(args.header)

boxsize   = cf.boxSize
redshift  = cf.redshift
scale_fac = 1./(1.+redshift)

OmegaM  = cf.Omega_M
OmegaDE = 1 - OmegaM
hubble  = cf.H0 / 100.0
Ez = np.sqrt(OmegaM * (1+redshift)**3 + OmegaDE)
dz_max = 0.15

run_parallel = True
#perc = False
Mmin = 10**12.5

halo_fname = in_path + halo_file
gal_fname = in_path + memgal_file


# from merge_richness_files import merge_richness_files
# merge_richness_files(phase, run_name, out_path, ofname_base)
# exit()


############################################
#### read in halos ####
f = h5py.File(halo_fname,'r')
halos = f['halos']
#print(halos.dtype)
mass = halos['mass'] # m200b
sel = (mass > Mmin)
x_halo = halos['x'][sel]
y_halo = halos['y'][sel]
z_halo = halos['z'][sel]
#rvir = halos['rvir'][sel]/1e3 # rvir is wrong in this file
mass = mass[sel]
gid = halos['gid'][sel] # use gid as halo id

index = np.argsort(-mass)
x_halo = x_halo[index]
y_halo = y_halo[index]
z_halo = z_halo[index]
mass = mass[index]
gid = gid[index]
print('finished reading halos')


#### read in galaxies ####
f = h5py.File(gal_fname,'r')
particles  = f['particles']
#print(particles.dtype)
x_gal = particles['x']
y_gal = particles['y']
z_gal = particles['z']
print('finished galaxies')

from scipy import spatial
rmax_tree = 2


class CalcRichness(object): # one pz slice at a time
    def __init__(self, pz_min, pz_max):
        self.pz_min = pz_min
        self.pz_max = pz_max
        d_pz = pz_max - pz_min

        # periodic boundary condition in pz direction
        if pz_min < depth: # near the lower boundary
            # for galaxies
            sel_pz1 = (z_gal < pz_max + 1.2*depth) 
            sel_pz2 = (z_gal > boxsize - 1.2*depth)
            x_gal_slice = np.append(x_gal[sel_pz1], x_gal[sel_pz2]) # oh well...
            y_gal_slice = np.append(y_gal[sel_pz1], y_gal[sel_pz2])
            z_gal_slice = np.append(z_gal[sel_pz1], z_gal[sel_pz2] - boxsize)

            # for halos: padding for percolation
            sel_pz1 = (z_halo < pz_max + 1.2*depth) 
            sel_pz2 = (z_halo > boxsize - 1.2*depth)
            self.x_halo_padded = np.append(x_halo[sel_pz1], x_halo[sel_pz2]) # oh well...
            self.y_halo_padded = np.append(y_halo[sel_pz1], y_halo[sel_pz2])
            self.z_halo_padded = np.append(z_halo[sel_pz1], z_halo[sel_pz2] - boxsize)
            self.mass_padded = np.append(mass[sel_pz1], mass[sel_pz2])
            self.gid_padded = np.append(gid[sel_pz1], gid[sel_pz2])

        elif pz_max > boxsize - depth: # near the upper boundary
            # for galaxies
            sel_pz1 = (z_gal > pz_min - 1.2*depth)
            sel_pz2 = (z_gal < 1.2*depth) 
            x_gal_slice = np.append(x_gal[sel_pz1], x_gal[sel_pz2])
            y_gal_slice = np.append(y_gal[sel_pz1], y_gal[sel_pz2])
            z_gal_slice = np.append(z_gal[sel_pz1], z_gal[sel_pz2] + boxsize)

            # for halos
            sel_pz1 = (z_halo > pz_min - 1.2*depth)
            sel_pz2 = (z_halo < 1.2*depth) 
            self.x_halo_padded = np.append(x_halo[sel_pz1], x_halo[sel_pz2])
            self.y_halo_padded = np.append(y_halo[sel_pz1], y_halo[sel_pz2])
            self.z_halo_padded = np.append(z_halo[sel_pz1], z_halo[sel_pz2] + boxsize)
            self.mass_padded = np.append(mass[sel_pz1], mass[sel_pz2])
            self.gid_padded = np.append(gid[sel_pz1], gid[sel_pz2])

        else: # safe in the middle
            sel_pz = (z_gal > pz_min - 1.2*depth)&(z_gal < pz_max + 1.2*depth)
            x_gal_slice = x_gal[sel_pz]
            y_gal_slice = y_gal[sel_pz]
            z_gal_slice = z_gal[sel_pz]

            # for halos
            sel_pz = (z_halo > pz_min - 1.2*depth)&(z_halo < pz_max + 1.2*depth)
            self.x_halo_padded = x_halo[sel_pz]
            self.y_halo_padded = y_halo[sel_pz]
            self.z_halo_padded = z_halo[sel_pz]
            self.mass_padded = mass[sel_pz]
            self.gid_padded = gid[sel_pz]

        # periodic boundary condition in x-y direction # only galaxies, no need to do this for halos
        x_all = []
        y_all = []
        z_all = []
        for x_pm in [-1, 0, 1]:
            for y_pm in [-1, 0, 1]:
                x_all.extend(x_gal_slice + x_pm * boxsize)
                y_all.extend(y_gal_slice + y_pm * boxsize)
                z_all.extend(z_gal_slice)

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        z_all = np.array(z_all)
        padding = 10
        sel = (x_all > -padding)&(x_all < boxsize + padding)&(y_all > -padding)&(y_all < boxsize + padding)

        self.x_gal = x_all[sel]
        self.y_gal = y_all[sel]
        self.z_gal = z_all[sel]

        gal_position = np.dstack([self.x_gal, self.y_gal])[0]
        gal_tree = spatial.cKDTree(gal_position)

        halo_position = np.dstack([self.x_halo_padded, self.y_halo_padded])[0]
        halo_tree = spatial.cKDTree(halo_position)

        self.indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)
        self.gal_taken = np.zeros(len(self.x_gal)) # for percolation

    def plot_check_pbc(self):
        plt.figure(figsize=(21,7))
        plt.subplot(131, aspect='equal')
        plt.scatter(self.x_gal[::1000], self.y_gal[::1000])
        plt.axhline(0)
        plt.axhline(boxsize)
        plt.axvline(0)
        plt.axvline(boxsize)

        plt.subplot(132, aspect='equal')
        plt.scatter(self.y_gal[::1000], self.z_gal[::1000])
        plt.axvline(0)
        plt.axvline(boxsize)
        plt.axhline(self.pz_min)
        plt.axhline(self.pz_max)

        plt.subplot(133, aspect='equal')
        plt.scatter(self.x_gal[::1000], self.z_gal[::1000])
        plt.axvline(0)
        plt.axvline(boxsize)
        plt.axhline(self.pz_min)
        plt.axhline(self.pz_max)

        plt.savefig('pbc.png')

    def get_richness(self, i_halo):
        gal_ind = self.indexes_tree[i_halo]
        x_cen = self.x_halo_padded[i_halo]
        y_cen = self.y_halo_padded[i_halo]
        z_cen = self.z_halo_padded[i_halo]

        # cut the LOS first!
        d = self.z_gal[gal_ind] - z_cen
        if use_cylinder == True: 
            sel_z = (np.abs(d) < depth)&(self.gal_taken[gal_ind] < 1e-4)
        elif use_pmem == True:
            dz = d / 3000. * Ez 
            sel_z = (np.abs(dz) < dz_max)&(self.gal_taken[gal_ind] < 1e-4) # TODO: probabilistic percolation
            dz = dz[sel_z]
        else:
            print('BUG!!')

        r = (self.x_gal[gal_ind][sel_z] - x_cen)**2 + (self.y_gal[gal_ind][sel_z] - y_cen)**2 
        r = np.sqrt(r)

        if use_rlambda == True:
            rlam_ini = 1
            rlam = rlam_ini
            for iteration in range(100):
                if use_cylinder == True:
                    ngal = len(r[r < rlam])
                elif use_pmem == True:
                    ngal = np.sum(pmem_weights(dz, r/rlam))
                else:
                    print('BUG!!')

                rlam_old = rlam
                rlam = (ngal/100.)**0.2 / scale_fac # phys -> comoving
                #print(rlam, rlam_old)
                if abs(rlam - rlam_old) < 1e-5:
                    break
        else:
            rlam = radius

        sel_mem = (r < rlam)&(self.gal_taken[gal_ind][sel_z] < 1e-4)
            
        if perc == True:
            self.gal_taken[np.array(gal_ind)[sel_z][sel_mem]] = 1
            # otherwise, all gal_taken elements remain zero
             
        if use_cylinder == True:
            lam = len(r[sel_mem])
        elif use_pmem == True:
            lam = np.sum(pmem_weights(dz, r/rlam))
        else:
            print('bug!!')
        #print(lam, len(self.gal_taken[self.gal_taken==1]))
        return rlam, lam

    def measure_richness(self):
        nh = len(self.x_halo_padded)
        print('nh =', nh)

        ofname = f'{out_path}/' + ofname_base + f'_{self.pz_min}_{self.pz_max}.dat'
        outfile = open(ofname, 'w')
        outfile.write('#gid, mass, px, py, pz, rlam, lam \n')
        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if self.z_halo_padded[ih] > self.pz_min and self.z_halo_padded[ih] < self.pz_max: # discard padded halos
                outfile.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.gid_padded[ih], self.mass_padded[ih], self.x_halo_padded[ih], self.y_halo_padded[ih], self.z_halo_padded[ih], rlam, lam))
        outfile.close()





def calc_one_bin(ibin):
    cr = CalcRichness(pz_min=ibin*100, pz_max=(ibin+1)*100)
    cr.measure_richness()





if __name__ == '__main__':
    
    if run_parallel == False:
        cr = CalcRichness(pz_min=0, pz_max=100)
        stop = timeit.default_timer()
        print('prep took', stop - start, 'seconds')
        
        start = stop
        cr.measure_richness()
        stop = timeit.default_timer()
        print('richness took', stop - start, 'seconds')

    # for i in [0]:#range(12):
    #     cr = CalcRichness(pz_min=i*100, pz_max=(i+1)*100)
    #     #cr.plot_check_pbc()
    #     


    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    
    if run_parallel == True:
        
        start = timeit.default_timer()
        # parallel
        from multiprocessing import Pool
        #n_job2 = int(boxsize/100.0)
        n_job2 = 20
        p = Pool(n_job2)
        p.map(calc_one_bin, range(n_job2))
        stop = timeit.default_timer()
        print('richness took', stop - start, 'seconds')
    
    # merge files
    from merge_richness_files import merge_richness_files
    #ofname_base = memgal_file.replace(".hdf5", "")
    merge_richness_files(out_path, ofname_base, boxsize)
