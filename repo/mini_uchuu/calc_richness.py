#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import pandas as pd
from scipy import spatial
import os, sys, glob
import fitsio
from astropy.io import fits
from multiprocessing import Pool
sys.path.append('../utils')
from read_yml import ReadYML
from periodic_boundary_condition import periodic_boundary_condition
from periodic_boundary_condition import periodic_boundary_condition_halos

yml_fname = sys.argv[1]
model_id = int(sys.argv[2])
depth = int(sys.argv[3])
#./calc_richness.py yml/mini_uchuu_grid.yml 0 30
# depth = -1 for pmem

Mmin = 10**12.5
n_parallel = 10
perc = True
use_rlambda = True

if depth > 0:
    use_cylinder = True
    use_pmem = False
    z_padding = 1.2 * depth
else:
    use_cylinder = False
    use_pmem = True
    from pmem_weights import pmem_weights
    z_padding = 400

rich_name = f'd{depth:.0f}'

para = ReadYML(yml_fname)
out_path = f'{para.output_loc}/model_{para.model_set}_{model_id}/'
if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')

# read in halos
if para.which_sim == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    rmu = ReadMiniUchuu()
    rmu.read_halos(Mmin)
    boxsize = rmu.boxsize
    OmegaM = rmu.OmegaM
    hubble = rmu.hubble
    hid_in = rmu.hid
    mass_in = rmu.M200m
    x_halo_in = rmu.xh
    y_halo_in = rmu.yh
    z_halo_in = rmu.zh

OmegaDE = 1 - OmegaM
Ez = np.sqrt(OmegaM * (1+para.redshift)**3 + OmegaDE)
dz_max = 0.15
scale_factor = 1/(1+para.redshift)

# read in galaxies
gal_fname = f'{out_path}/gals.fit'
data, header = fitsio.read(gal_fname, header=True)
x_gal_in = data['px']
y_gal_in = data['py']
z_gal_in = data['pz']

#### periodic boundary condition ####
x_padding = 3
y_padding = 3

x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(x_halo_in, y_halo_in, z_halo_in, 
    boxsize, x_padding, y_padding, z_padding, hid_in, mass_in)

sort = np.argsort(-mass)
hid = hid[sort]
mass = mass[sort]
x_halo = x_halo[sort]
y_halo = y_halo[sort]
z_halo = z_halo[sort]

x_gal, y_gal, z_gal = periodic_boundary_condition(x_gal_in, y_gal_in, z_gal_in,
    boxsize, x_padding, y_padding, z_padding)

class CalcRichness(object): # one pz slice at a time
    def __init__(self, pz_min, pz_max):
        self.pz_min = pz_min
        self.pz_max = pz_max

        sel_gal = (z_gal > pz_min - z_padding) & (z_gal < pz_max + z_padding)
        self.x_gal = x_gal[sel_gal]
        self.y_gal = y_gal[sel_gal]
        self.z_gal = z_gal[sel_gal]
        
        sel_halo = (z_halo > pz_min - z_padding) & (z_halo < pz_max + z_padding)
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
                rlam = (ngal/100.)**0.2 / scale_factor # phys -> comoving
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
        nh = len(self.x_halo)
        print('nh =', nh)

        ofname = f'{out_path}/temp/richness_{rich_name}_pz{self.pz_min:.0f}_{self.pz_max:.0f}.dat'
        outfile = open(ofname, 'w')
        outfile.write('#hid, mass, px, py, pz, rlam, lam \n')
        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if self.z_halo[ih] > self.pz_min and self.z_halo[ih] < self.pz_max and \
            0 < self.x_halo[ih] < boxsize and 0 < self.y_halo[ih] < boxsize: # discard padded halos
                outfile.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.hid[ih], self.mass[ih], self.x_halo[ih], self.y_halo[ih], self.z_halo[ih], rlam, lam))
        outfile.close()



layer_thickness = boxsize / n_parallel

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

    os.system(f'rm -rf {out_path}/temp/richness_{rich_name}_pz*.dat')


if __name__ == '__main__':
    stop = timeit.default_timer()
    print('prep took', stop - start, 'seconds')
    start = timeit.default_timer()
    p = Pool(n_parallel)
    p.map(calc_one_bin, range(n_parallel))
    stop = timeit.default_timer()
    print('richness took', stop - start, 'seconds')
    
    start = stop
    merge_files()
    stop = timeit.default_timer()
    print('merging took', stop - start, 'seconds')
