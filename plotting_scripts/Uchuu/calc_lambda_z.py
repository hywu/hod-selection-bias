import fitsio
import numpy as np
import scipy
import copy
import time
import astropy
import yaml
import os
import sys
from astropy import cosmology
from astropy.io import fits
import astropy.units as u
import astropy.cosmology.units as cu
from multiprocessing import Pool

cat_name = sys.argv[1] #gauss90_scan

n_parallel_x = 5
n_parallel_y = 5
n_parallel_z = 10
n_parallel = n_parallel_x * n_parallel_y * n_parallel_z

Mmin = 5e12
boxsize = 2000.0 #Mpc/h
total_volume = boxsize**3

halo_fname = 'host_halos_047_M12.5.fit'
data, header = fitsio.read(halo_fname, header=True)
mass = data['M200m']
sel = (mass > Mmin)
x_halo = data['px'][sel]
y_halo = data['py'][sel]
z_halo = data['pz'][sel]
haloid = data['haloid'][sel]

#cosmo parameters assumed by Myles
H0_astropy = 67.74 * u.km/u.s/u.Mpc
Om_astropy = 0.3089
cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0_astropy, Om0=Om_astropy)

comoving_r = cosmo.comoving_distance(np.array([0.08,0.12]))
comoving_r = comoving_r.to(u.Mpc/cu.littleh, cu.with_H0(H0_astropy))
comoving_r = comoving_r.value

deg2 = 10400.288
strad = deg2 * ((np.pi/180.0)**2)
comoving_volume_myles = (strad/3) * (comoving_r[1]**3 - comoving_r[0]**3)

data = np.loadtxt('myles_data.dat')
# f_proj_myles = data[:,0]
# b_lambda_myles = data[:,1]
n_bin_cl_myles = data[:,2]
# mean_bin_richness_myles = data[:,3]
# myles_f_proj_err = np.array([0.007, 0.021, 0.025, 0.045, 0.024, 0.025])
subvolume = total_volume / (n_parallel_x * n_parallel_y * n_parallel_z)
volume_ratio = subvolume / comoving_volume_myles

abundance_match_n = volume_ratio * n_bin_cl_myles
abundance_match_n = np.rint(abundance_match_n)
abundance_match_n = int(np.sum(abundance_match_n))

loc = f'model_fid_hod_z0.1/{cat_name}/'

def calc_lambda_z(px_min, px_max, py_min, py_max, pz_min, pz_max):
    #selecting halos within the coordinate bounds of the bin
    sel = (px_min <= x_halo) & (x_halo <= px_max) & (py_min <= y_halo) & (y_halo <= py_max) & (pz_min <= z_halo) & (z_halo <= pz_max)
    sel_haloid = haloid[sel]
    
    dz_scan_list = np.arange(0, 201, 5)
    w_master_list = []
    
    #reading in fiducial richness
    fname = loc + f'richness_{cat_name}0.fit'
    data, header = fitsio.read(fname, header=True)
    lam_fid = data['lambda']
    haloid_fid = data['haloid']
    
    #selecting halos in richness catalog within coordinate bounds
    sel_coord = np.isin(haloid_fid, sel_haloid)
    lam_fid = lam_fid[sel_coord]
    haloid_fid = haloid_fid[sel_coord]
    
    #abundance matching
    sel = np.argsort(lam_fid)
    lam_fid = lam_fid[sel]
    haloid_fid = haloid_fid[sel]
    
    lam_fid = lam_fid[-abundance_match_n:]
    haloid_fid = haloid_fid[-abundance_match_n:]
    lam_fid = np.mean(lam_fid)
    
    for dz_scan in dz_scan_list:
        fname = loc + f'richness_{cat_name}{dz_scan}.fit'
        data, header = fitsio.read(fname, header=True)
        lam_w = data['lambda']
        haloid_w = data['haloid']
        
        sel = np.isin(haloid_w, haloid_fid)
        w_master_list.append(np.mean(lam_w[sel])/lam_fid)
        
    return w_master_list

z_thickness = boxsize / n_parallel_z
x_thickness = boxsize / n_parallel_x
y_thickness = boxsize / n_parallel_y

def calc_one_bin(ibin):
    iz = ibin // (n_parallel_x * n_parallel_y)
    ixy = ibin % (n_parallel_x * n_parallel_y)
    ix = ixy // n_parallel_x
    iy = ixy % n_parallel_x
    
    pz_min = iz*z_thickness
    pz_max = (iz+1)*z_thickness
    
    px_min = ix*x_thickness
    px_max = (ix+1)*x_thickness
    
    py_min = iy*y_thickness
    py_max = (iy+1)*y_thickness
#     print(calc_dbl_gauss_param(px_min, px_max, py_min, py_max, pz_min, pz_max))
    return calc_lambda_z(px_min, px_max, py_min, py_max, pz_min, pz_max)

if __name__=='__main__':
    start = time.time()
    p = Pool(processes=28)
    w_all_bin_list = p.map(calc_one_bin, np.arange(0,n_parallel,1))
    np.savetxt(loc+'w_'+cat_name+'.dat', np.transpose(w_all_bin_list), '%g', header='Column is param for 1 subvolume')
    stop = time.time()
    print(f'Total time: {(stop-start)/3600.0} hours')