#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import matplotlib.pyplot as plt
import fitsio, h5py
import os, sys, glob
import pandas as pd
import argparse
from astropy.io import fits
from multiprocessing import Pool
from scipy import spatial

parser = argparse.ArgumentParser()
parser.add_argument('--which_sim', type=str, required=True, help='')
parser.add_argument('--model_id', type=int, required=True, help='')
parser.add_argument('--depth', type=float, required=True, help='')
args = parser.parse_args()
#./calc_richness.py --which_sim mini_uchuu --model_id 0 --depth 30

if args.which_sim == 'mini_uchuu':
    output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'

out_path = f'{output_loc}/model_{args.model_id}/'

use_rlambda = True
perc = True
Mmin = 10**12.5
use_cylinder = True
use_pmem = False

# if args.depth > 0:
#     use_cylinder = True
#     use_pmem = False
# if args.depth == -1 : # BUGGY!
#     use_cylinder = False
#     use_pmem = True

depth = args.depth
rich_name = f'd{depth:.0f}'

if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')


#### things can be moved to yml files ####
redshift = 0.3
scale_fac = 1/(1+redshift)

############################################
#### read in halos ####
# input_loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
# data = h5py.File(input_loc+f'MiniUchuu_halolist_z0p30.h5', 'r')
# hid = np.array(data['id'])
# pid = np.array(data['pid'])
# sel1 = (pid == -1)
# M200m = np.array(data['M200b'])[sel1]
# sel2 = (M200m > Mmin)
# mass = M200m[sel2]
# gid = hid[sel1][sel2]
# x_halo = np.array(data['x'])[sel1][sel2]
# y_halo = np.array(data['y'])[sel1][sel2]
# z_halo = np.array(data['z'])[sel1][sel2]
# print('finished reading halos')

if args.which_sim == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    rmu = ReadMiniUchuu()
    rmu.read_halos(Mmin)
    boxsize = rmu.boxsize
    x_halo = rmu.xh
    y_halo = rmu.yh
    z_halo = rmu.zh
    mass = rmu.M200m
    gid = rmu.gid
    OmegaM = rmu.OmegaM
    hubble = rmu.hubble

#hubble = 0.6774
#OmegaM = 0.3089
OmegaDE = 1 - OmegaM
Ez = np.sqrt(OmegaM * (1+redshift)**3 + OmegaDE)
dz_max = 0.15


gal_fname = f'{out_path}/gals.fit'
data, header = fitsio.read(gal_fname, header=True)
x_gal = np.array(data['px'])
y_gal = np.array(data['py'])
z_gal = np.array(data['pz'])

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

        rmax_tree = 2
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

    def get_ngal(self, R):
        gal_ind = self.indexes_tree[i_halo]
        x_cen = self.x_halo_padded[i_halo]
        y_cen = self.y_halo_padded[i_halo]
        z_cen = self.z_halo_padded[i_halo]
        r = (self.x_gal[gal_ind][sel_z] - x_cen)**2 
        r += (self.y_gal[gal_ind][sel_z] - y_cen)**2 
        r += (self.z_gal[gal_ind][sel_z] - z_cen)**2 
        r = np.sqrt(r)        
        ngal = len(r[r < R])
        return ngal
        
        
    def get_richness(self, i_halo):
        gal_ind = self.indexes_tree[i_halo]
        x_cen = self.x_halo_padded[i_halo]
        y_cen = self.y_halo_padded[i_halo]
        z_cen = self.z_halo_padded[i_halo]
        
        # cut the LOS first!
        d = self.z_gal[gal_ind] - z_cen
        if use_cylinder == True and depth > 0: 
            sel_z = (np.abs(d) < depth)&(self.gal_taken[gal_ind] < 1e-4)
        elif use_pmem == True or depth == -1:
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
                if use_cylinder == True and depth > 0:
                    ngal = len(r[r < rlam])
                elif use_pmem == True or depth == -1:
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
             
        if use_cylinder == True and depth > 0:
            lam = len(r[sel_mem])
        elif use_pmem == True or depth == -1:
            lam = np.sum(pmem_weights(dz, r/rlam))
        else:
            print('bug!!')
        #print(lam, len(self.gal_taken[self.gal_taken==1]))
        return rlam, lam

    def measure_richness(self):
        nh = len(self.x_halo_padded)
        print('nh =', nh)

        ofname = f'{out_path}/temp/richness_{rich_name}_pz{self.pz_min}_{self.pz_max}.dat'
        outfile = open(ofname, 'w')
        outfile.write('#gid, mass, px, py, pz, rlam, lam \n')
        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if self.z_halo_padded[ih] > self.pz_min and self.z_halo_padded[ih] < self.pz_max: # discard padded halos
                outfile.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.gid_padded[ih], self.mass_padded[ih], self.x_halo_padded[ih], self.y_halo_padded[ih], self.z_halo_padded[ih], rlam, lam))
        outfile.close()


layer_thickness = 40
nlayer = 10
def calc_one_bin(ibin):
    cr = CalcRichness(pz_min=ibin*layer_thickness, pz_max=(ibin+1)*layer_thickness)
    cr.measure_richness()


    


def merge_files():
    fname_list = glob.glob(f'{out_path}/temp/richness_{rich_name}_pz*.dat')
    print('nfiles', len(fname_list))
    hid_out = []
    m_out = []
    x_out = []
    y_out = []
    z_out = []
    rlam_out = []
    lam_out = []

    for fname in fname_list:
        data = pd.read_csv(fname, delim_whitespace=True, dtype=np.float64, comment='#', 
                        names=['haloid', 'mass', 'px', 'py', 'pz', 'rlam', 'lam'])
        hid_out.extend(data['haloid'])
        m_out.extend(data['mass'])
        x_out.extend(data['px'])
        y_out.extend(data['py'])
        z_out.extend(data['pz'])
        rlam_out.extend(data['rlam'])
        lam_out.extend(data['lam'])
        
    hid_out = np.array(hid_out)
    m_out = np.array(m_out)
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    z_out = np.array(z_out)
    rlam_out = np.array(rlam_out)
    lam_out = np.array(lam_out)

    sel = np.argsort(-m_out)

        
    cols=[
      fits.Column(name='haloid', format='K' ,array=hid_out[sel]),
      fits.Column(name='M200m', format='E',array=m_out[sel]),
      fits.Column(name='px', format='D' ,array=x_out[sel]),
      fits.Column(name='py', format='D',array=y_out[sel]),
      fits.Column(name='pz', format='D',array=z_out[sel]),
      fits.Column(name='Rlamda', format='D',array=rlam_out[sel]),
      fits.Column(name='lambda', format='D',array=lam_out[sel]),
    ]
    coldefs = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    tbhdu.writeto(f'{out_path}/richness_{rich_name}.fit', overwrite=True)



if __name__ == '__main__':
    run_parallel = True

    # if run_parallel == False:
    #     cr = CalcRichness(pz_min=0, pz_max=layer_thickness)
    #     stop = timeit.default_timer()
    #     print('prep took', stop - start, 'seconds')
    #     start = stop
    #     cr.measure_richness()
    #     stop = timeit.default_timer()
    #     print('richness took', stop - start, 'seconds')

    # stop = timeit.default_timer()
    # print('prep took', stop - start, 'seconds')
    
    if run_parallel == True:
        start = timeit.default_timer()
        p = Pool(nlayer)
        p.map(calc_one_bin, range(nlayer))
        stop = timeit.default_timer()
        print('richness took', stop - start, 'seconds')
    
    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', stop - start, 'seconds')
