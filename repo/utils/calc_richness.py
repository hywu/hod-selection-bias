#!/usr/bin/env python
import timeit
start = timeit.default_timer()
import numpy as np
import pandas as pd
from scipy import spatial
import os
import sys
import glob
import fitsio
import yaml
from astropy.io import fits
from multiprocessing import Pool
sys.path.append('../utils')
#from read_yml import ReadYML
from periodic_boundary_condition import periodic_boundary_condition
from periodic_boundary_condition import periodic_boundary_condition_halos

yml_fname = sys.argv[1]
#./calc_richness.py yml/mini_uchuu_fid_hod.yml
#./calc_richness.py yml/abacus_summit_fid_hod.yml

#model_id = 0 # int(sys.argv[2])
 #int(sys.argv[3])
#./calc_richness.py yml/mini_uchuu_grid.yml 0 30
# depth = -1 for pmem

# NOTE! pmem not working yet

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

depth = para['depth']
perc = para['perc']#True #False
use_rlambda = para['use_rlambda'] #True #False
#radius = 1 

Mmin = 10**12.5

n_parallel_z = 1 # NOTE! cannot do more than one yet.
n_parallel_x = 10
n_parallel_y = 10

n_parallel = n_parallel_z * n_parallel_x * n_parallel_y


if depth > 0:
    use_cylinder = True
    use_pmem = False
    #z_padding = 1.2 * depth
else:
    use_cylinder = False
    use_pmem = True
    from pmem_weights import pmem_weights
    z_padding = 400


z_padding_halo = 0
z_padding_gal = 0

rich_name = f'd{depth:.0f}'

# para = ReadYML(yml_fname)
# out_path = f'{para.output_loc}/model_{para.model_set}_{model_id}/'
output_loc = para['output_loc']
model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}'
if os.path.isdir(out_path)==False: os.makedirs(out_path)
if os.path.isdir(out_path+'/temp/')==False: os.makedirs(out_path+'/temp/')

# read in halos
if para['nbody'] == 'mini_uchuu':
    from read_mini_uchuu import ReadMiniUchuu
    readcat = ReadMiniUchuu(para['nbody_loc'])
    readcat.read_halos(Mmin)
    boxsize = readcat.boxsize
    OmegaM = readcat.OmegaM
    hubble = readcat.hubble
    hid_in = readcat.hid
    mass_in = readcat.M200m
    x_halo_in = readcat.xh
    y_halo_in = readcat.yh
    z_halo_in = readcat.zh

if para['nbody'] == 'abacus_summit':
    sys.path.append('../abacus_summit')
    from read_abacus_summit import ReadAbacusSummit
    readcat = ReadAbacusSummit(para['nbody_loc'])
    readcat.read_halos(Mmin)
    boxsize = readcat.boxsize
    OmegaM = readcat.OmegaM
    hubble = readcat.hubble
    hid_in = readcat.hid
    mass_in = readcat.mass # sigh, this is actually Mvir
    x_halo_in = readcat.xh
    y_halo_in = readcat.yh
    z_halo_in = readcat.zh

OmegaDE = 1 - OmegaM
Ez = np.sqrt(OmegaM * (1+para['redshift'])**3 + OmegaDE)
dz_max = 0.15
scale_factor = 1/(1+para['redshift'])

# read in galaxies
gal_fname = f'{out_path}/gals.fit'
data, header = fitsio.read(gal_fname, header=True)
x_gal_in = data['px']
y_gal_in = data['py']
z_gal_in = data['pz']

#### periodic boundary condition ####
x_padding = 3
y_padding = 3

x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(
    x_halo_in, y_halo_in, z_halo_in, 
    boxsize, x_padding, y_padding, 0, hid_in, mass_in)
#x_halo, y_halo, z_halo, hid, mass = periodic_boundary_condition_halos(x_halo_in, y_halo_in, z_halo_in, 
#    boxsize, 400, 400, 400, hid_in, mass_in)

sort = np.argsort(-mass)
hid = hid[sort]
mass = mass[sort]
x_halo = x_halo[sort]
y_halo = y_halo[sort]
z_halo = z_halo[sort]

x_gal, y_gal, z_gal = periodic_boundary_condition(x_gal_in, y_gal_in, z_gal_in,
    boxsize, x_padding, y_padding, 0)
#x_gal, y_gal, z_gal = periodic_boundary_condition(x_gal_in, y_gal_in, z_gal_in,
#    boxsize, 400, 400, 400)

class CalcRichness(object): # one pz slice at a time
    def __init__(self, pz_min, pz_max, px_min=0, px_max=boxsize, py_min=0, py_max=boxsize):
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

        # sort = np.argsort(-self.mass)
        # self.hid = self.hid[sort]
        # self.mass = self.mass[sort]
        # self.x_halo = self.x_halo[sort]
        # self.y_halo = self.y_halo[sort]
        # self.z_halo = self.z_halo[sort]

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
        #d = self.z_gal[gal_ind] - z_cen
        z_gal_gal_ind = self.z_gal[gal_ind]
        d_pbc0 = z_gal_gal_ind - z_cen
        d_pbc1 = z_gal_gal_ind + boxsize - z_cen
        d_pbc2 = z_gal_gal_ind - boxsize - z_cen


        if use_cylinder == True and depth > 0: 
            #sel_z = (np.abs(d) < depth)&(self.gal_taken[gal_ind] < 1e-4)
            sel_z0 = (np.abs(d_pbc0) < depth)
            sel_z1 = (np.abs(d_pbc1) < depth)
            sel_z2 = (np.abs(d_pbc2) < depth)
            sel_z = sel_z0 | sel_z1 | sel_z2
            sel_z = sel_z & (self.gal_taken[gal_ind] < 1e-4)


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

        sel_mem = (r < rlam)#&(self.gal_taken[gal_ind][sel_z] < 1e-4)
            
        if perc == True and len(gal_ind) > 0:
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

        ofname = f'{out_path}/temp/richness_{rich_name}_pz{self.pz_min:.0f}_{self.pz_max:.0f}_px{self.px_min:.0f}_{self.px_max:.0f}_py{self.py_min:.0f}_{self.py_max:.0f}.dat'
        outfile = open(ofname, 'w')
        outfile.write('#hid, mass, px, py, pz, rlam, lam \n')
        for ih in range(nh):
            rlam, lam = self.get_richness(ih)
            if self.z_halo[ih] > self.pz_min and self.z_halo[ih] < self.pz_max and \
                self.x_halo[ih] > self.px_min and self.x_halo[ih] < self.px_max and \
                self.y_halo[ih] > self.py_min and self.y_halo[ih] < self.py_max:
                #0 < self.x_halo[ih] < boxsize and 0 < self.y_halo[ih] < boxsize: # discard padded halos
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
    #print(iz, ix, iy)
    cr = CalcRichness(pz_min=iz*z_layer_thickness, pz_max=(iz+1)*z_layer_thickness,
        px_min=ix*x_cube_size, px_max=(ix+1)*x_cube_size,
        py_min=iy*y_cube_size, py_max=(iy+1)*y_cube_size
        )
    #cr = CalcRichness(pz_min=ibin*z_layer_thickness, pz_max=(ibin+1)*z_layer_thickness)
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

    cols = [
        fits.Column(name='haloid', format='K' ,array=hid_out[sel]),
        fits.Column(name='M200m', format='E',array=m_out[sel]),
        fits.Column(name='px', format='D' ,array=x_out[sel]),
        fits.Column(name='py', format='D',array=y_out[sel]),
        fits.Column(name='pz', format='D',array=z_out[sel]),
        fits.Column(name='Rlambda', format='D',array=rlam_out[sel]),
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
    
