#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys
from distutils import util
from multiprocessing import Pool
from scipy import spatial
import argparse
import configparser

import config
path_to_here = str(os.path.dirname(os.path.realpath(__file__)))
path_to_utils = path_to_here.replace('richness', 'utils')
sys.path.append(path_to_utils)
print(sys.path)
from periodic_boundary_condition import periodic_boundary_condition
from periodic_boundary_condition import periodic_boundary_condition_halos

from pmem_weights import pmem_weights
from pmem_weights import pmem_quad_top_hat

'''
## example (output to Heidi's space)
./calc_richness.py --halos "halo_cut3.00e+12_base_c000_ph000_z0p300.h5" --members "NHOD_0.10_11.7_11.7_12.9_1.00_0.0_0.0_1.0_1.0_0.0_c000_ph000_z0p300.hdf5" --header "/fs/project/PAS0023/Snapshots/AbacusSummit_base/base_c000/base_c000_ph000/z0p300/header" --use_cylinder --depth 10 --input_path "/fs/project/PAS0023/Snapshots/AbacusSummit_base/base_c000/base_c000_ph000/z0p300/" --output_path "/fs/scratch/PCON0003/cond0099/test_summit/" --ID_str "NHOD_0.10_11.7_11.7_12.9_1.00_0.0_0.0_1.0_1.0_0.0_c000_ph000_z0p300"
'''

## required
parser = argparse.ArgumentParser()
parser.add_argument('--halos', required=True, help='File name of halo hdf5 file')
parser.add_argument('--members', required=True, help='File name of mock member galaxies hdf5 file')
parser.add_argument('--header', required=True, help='Header file name for simulation')

group = parser.add_mutually_exclusive_group() # cylinder or pmem?
group.add_argument('--use_cylinder', action='store_true', help='choose either use_cylidner or use_pmem or use_quad_top_hat')
group.add_argument('--use_pmem', action='store_true', help='choose either use_cylinder or use_pmem or use_quad_top_hat')
group.add_argument('--use_quad_top_hat', action='store_true', help='choose either use_cylinder or use_pmem or use_quad_top_hat')

## optional
parser.add_argument('--depth', type=float, help='required if use_cylinder==True or use_quad_top_hat==True') # 30

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
use_quad_top_hat = args.use_quad_top_hat

## cylinder or pmem or quad_top?
if args.use_quad_top_hat == True:
    depth = args.depth
    print('using quad_top_hat with depth (full width at minimum)', depth)

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
    ofname_base  += '_d'+'{:.2f}'.format(depth)
if use_quad_top_hat == True:
    ofname_base  += '_quad_d'+'{:.2f}'.format(depth)
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
    os.makedirs(f'{out_path}/temp/')

print('input path', in_path)
print('output path', out_path)


#### things can be moved to yml files ####
try:
    myconfigparser = configparser.ConfigParser()
    myconfigparser.read(args.header)
    params = myconfigparser['params']
    boxsize   = float(params['boxsize'])
    redshift  = float(params['redshift'])
    OmegaM  = float(params['Omega_M'])
    hubble  = float(params['h'])
            
except:
    cf = config.AbacusConfigFile(args.header)
    boxsize   = cf.boxSize
    redshift  = cf.redshift
    OmegaM  = cf.Omega_M
    hubble  = cf.H0 / 100.0

scale_fac = 1./(1.+redshift)
OmegaDE = 1 - OmegaM
Ez = np.sqrt(OmegaM * (1+redshift)**3 + OmegaDE)
dz_max = 0.15

run_parallel = True
#perc = False
Mmin = 10**12.5

halo_fname = in_path + halo_file
gal_fname = out_path + memgal_file


############################################
#### read in halos ####
f = h5py.File(halo_fname,'r')
halos = f['halos']
#print(halos.dtype)
try:
    hid_in = halos['gid']
except:
    hid_in = np.arange(0, len(halos['x']))    
mass_in = halos['mass']
x_halo_in = halos['x']
y_halo_in = halos['y']
z_halo_in = halos['z']
print('finished reading halos')


#### read in galaxies ####
f = h5py.File(gal_fname,'r')
particles  = f['particles']
#print(particles.dtype)
x_gal_in = particles['x']
y_gal_in = particles['y']
z_gal_in = particles['z']
print('finished galaxies')


n_parallel_z = 1 # NOTE! cannot do more than one yet.
n_parallel_x = int(boxsize/100.)
n_parallel_y = int(boxsize/100.)

n_parallel = n_parallel_z * n_parallel_x * n_parallel_y



rmax_tree = 2


#### periodic boundary condition ####
x_padding = 3
y_padding = 3
z_padding_halo = 0  ## fully periodic 
z_padding_gal = 0  ## fully periodic

x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(
    x_halo_in, y_halo_in, z_halo_in, 
    boxsize, x_padding, y_padding, 0, hid_in, mass_in)

sort = np.argsort(-mass)
hid = hid[sort]
mass = mass[sort]
x_halo = x_halo[sort]
y_halo = y_halo[sort]
z_halo = z_halo[sort]

x_gal, y_gal, z_gal = periodic_boundary_condition(
    x_gal_in, y_gal_in, z_gal_in,
    boxsize, x_padding, y_padding, 0)


class CalcRichness(object):
    def __init__(self, pz_min=0, pz_max=boxsize, px_min=0, px_max=boxsize, py_min=0, py_max=boxsize):
        self.pz_min = pz_min
        self.pz_max = pz_max
        self.px_min = px_min
        self.px_max = px_max
        self.py_min = py_min
        self.py_max = py_max

        sel_gal = (z_gal > pz_min - z_padding_gal) & (z_gal < pz_max + z_padding_gal)
        if px_min > 0 or px_max < boxsize or py_min > 0 or py_max < boxsize:
            sel_gal &= (x_gal > px_min - x_padding) & (x_gal < px_max + x_padding)
            sel_gal &= (y_gal > py_min - y_padding) & (y_gal < py_max + y_padding)

        self.x_gal = x_gal[sel_gal]
        self.y_gal = y_gal[sel_gal]
        self.z_gal = z_gal[sel_gal]
        
        sel_halo = (z_halo > pz_min - z_padding_halo) & (z_halo < pz_max + z_padding_halo)
        if px_min > 0 or px_max < boxsize or py_min > 0 or py_max < boxsize:
            sel_halo &= (x_halo > px_min - x_padding) & (x_halo < px_max + x_padding)
            sel_halo &= (y_halo > py_min - y_padding) & (y_halo < py_max + y_padding)

        self.x_halo = x_halo[sel_halo]
        self.y_halo = y_halo[sel_halo]
        self.z_halo = z_halo[sel_halo]
        self.hid = hid[sel_halo]
        self.mass = mass[sel_halo]

        gal_position = np.dstack([self.x_gal, self.y_gal])[0]
        gal_tree = spatial.cKDTree(gal_position)

        halo_position = np.dstack([self.x_halo, self.y_halo])[0]
        halo_tree = spatial.cKDTree(halo_position)

        rmax_tree = 2
        self.indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)
        self.gal_taken = np.zeros(len(self.x_gal)) # for percolation

    def get_richness(self, i_halo):
        gal_ind = self.indexes_tree[i_halo]
        x_cen = self.x_halo[i_halo]
        y_cen = self.y_halo[i_halo]
        z_cen = self.z_halo[i_halo]

        # cut the LOS first!
        z_gal_gal_ind = self.z_gal[gal_ind]
        d_pbc0 = z_gal_gal_ind - z_cen
        d_pbc1 = z_gal_gal_ind + boxsize - z_cen
        d_pbc2 = z_gal_gal_ind - boxsize - z_cen

        if use_cylinder == True and depth > 0: 
            sel_z0 = (np.abs(d_pbc0) < depth)
            sel_z1 = (np.abs(d_pbc1) < depth)
            sel_z2 = (np.abs(d_pbc2) < depth)
            sel_z = sel_z0 | sel_z1 | sel_z2
            sel_z = sel_z & (self.gal_taken[gal_ind] < 1e-4)

        elif use_pmem == True or depth == -1:
            dz = d / 3000. * Ez 
            sel_z = (np.abs(dz) < dz_max)&(self.gal_taken[gal_ind] < 1e-4) # TODO: probabilistic percolation
            dz = dz[sel_z]
        
        elif use_quad_top_hat==True and depth>0:
            sel_z0 = (np.abs(d_pbc0) < depth)
            sel_z1 = (np.abs(d_pbc1) < depth)
            sel_z2 = (np.abs(d_pbc2) < depth)
            sel_z = sel_z0 | sel_z1 | sel_z2
            sel_z = sel_z & (self.gal_taken[gal_ind] < 1e-4)
            d_pbc0 = d_pbc0[sel_z]
            d_pbc1 = d_pbc1[sel_z]
            d_pbc2 = d_pbc2[sel_z]
            
        else:
            print('BUG!!')

        r = (self.x_gal[gal_ind][sel_z] - x_cen)**2 + (self.y_gal[gal_ind][sel_z] - y_cen)**2 
        r = np.sqrt(r)

        if use_rlambda == True:
            rlam_ini = 1
            rlam = rlam_ini
            for iteration in range(100):
                if use_cylinder == True and depth > 0:
                    ngal = len(r[r < rlam])
                elif use_pmem == True or depth == -1:
                    ngal = np.sum(pmem_weights(dz, r/rlam))
                elif use_quad_top_hat==True and depth>0:
                    ngal = np.sum( pmem_quad_top_hat(d_pbc0[r<rlam], depth) + pmem_quad_top_hat(d_pbc1[r<rlam], depth) + pmem_quad_top_hat(d_pbc2[r<rlam], depth))
                else:
                    print('BUG!!')

                rlam_old = rlam
                rlam = (ngal/100.)**0.2 / scale_fac # phys -> comoving
                if abs(rlam - rlam_old) < 1e-5:
                    break
        else:
            rlam = radius

        sel_mem = (r < rlam)#&(self.gal_taken[gal_ind][sel_z] < 1e-4)
            
        if perc == True and len(gal_ind) > 0:
            self.gal_taken[np.array(gal_ind)[sel_z][sel_mem]] = 1
            # otherwise, all gal_taken elements remain zero
             
        if use_cylinder == True and depth > 0:
            lam = len(r[sel_mem])
            print("Using Uniform Top-Hat...")
        elif use_pmem == True or depth == -1:
            lam = np.sum(pmem_weights(dz, r/rlam))
            print("Using interpolated 2D-Pmem distribution...")
        elif use_quad_top_hat==True and depth>0:
            lam = np.sum( pmem_quad_top_hat(d_pbc0[sel_mem], depth) + pmem_quad_top_hat(d_pbc1[sel_mem], depth) + pmem_quad_top_hat(d_pbc2[sel_mem], depth))
            print("Using Quadratic Top-Hat...")
        else:
            print('bug!!')
        #print(lam, len(self.gal_taken[self.gal_taken==1]))
        return rlam, lam

    def measure_richness(self):
        nh = len(self.x_halo)
        #print('nh =', nh)

        if os.path.isdir(out_path)==False:
            ofname = f'{out_path}/temp/{ofname_base}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
        else:
            ofname = f'{out_path}/{ofname_base}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
            
        outfile = open(ofname, 'w')
        outfile.write('#hid, mass, px, py, pz, rlam, lam \n')
        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if self.z_halo[ih] > self.pz_min and self.z_halo[ih] < self.pz_max and \
                self.x_halo[ih] > self.px_min and self.x_halo[ih] < self.px_max and \
                self.y_halo[ih] > self.py_min and self.y_halo[ih] < self.py_max:
                outfile.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.hid[ih], self.mass[ih], self.x_halo[ih], self.y_halo[ih], self.z_halo[ih], rlam, lam))
        outfile.close()


z_layer_thickness = boxsize / n_parallel_z
x_cube_size = boxsize / n_parallel_x
y_cube_size = boxsize / n_parallel_y


def calc_one_bin(ibin):
    iz = ibin // (n_parallel_x * n_parallel_y)
    ixy = ibin % (n_parallel_x * n_parallel_y)
    ix = ixy // n_parallel_x
    iy = ixy % n_parallel_x
    cr = CalcRichness(pz_min=iz*z_layer_thickness, pz_max=(iz+1)*z_layer_thickness,
        px_min=ix*x_cube_size, px_max=(ix+1)*x_cube_size,
        py_min=iy*y_cube_size, py_max=(iy+1)*y_cube_size
        )
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


    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    
    if run_parallel == True:
        
        start = timeit.default_timer()
        # parallel
        p = Pool(n_parallel)
        p.map(calc_one_bin, range(n_parallel))
        stop = timeit.default_timer()
        print('richness took', stop - start, 'seconds')
    
    # merge files
    from merge_richness_files import merge_richness_files
    #ofname_base = memgal_file.replace(".hdf5", "")
    merge_richness_files(out_path, ofname_base, boxsize)
