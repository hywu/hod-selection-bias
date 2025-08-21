import fitsio
import numpy as np
import scipy
import copy
import time
import astropy
import yaml
import os
from astropy import cosmology
from astropy.io import fits
import astropy.units as u
import astropy.cosmology.units as cu
from kllr import kllr_model
from multiprocessing import Pool

#is it good practice to unappend the path?
import sys
sys.path.append('/home/andy/Documents/hod-selection-bias/repo/utils/')


yml_loc = "/home/andy/Documents/hod-selection-bias/repo/utils/yml/"
yml_fname_list = ['uchuu_fid_hod.yml']
yml_fname = yml_loc + yml_fname_list[0]


cat_name = sys.argv[1] #quad200
cat_name = '_' + cat_name + '.fit'

print(cat_name)

n_parallel_x = 5
n_parallel_y = 5
n_parallel_z = 10
n_parallel = n_parallel_x * n_parallel_y * n_parallel_z

Mmin = 5e12
boxsize = 2000.0 #Mpc/h
total_volume = boxsize**3

#reading parameters from yml file
with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

nbody = para['nbody']
OmegaM = para['OmegaM']
OmegaL = para['OmegaL']
h = para['hubble']
sigma8 = para['sigma8']
OmegaB = para['OmegaB']
ns = para['ns']
model_name = para['model_name']
redshift = para['redshift']
alpha = para['alpha']
lgM20 = para.get('lgM20', None)
lgkappa = para['lgkappa']
lgMcut = para['lgMcut']
sigmalogM = para['sigmalogM']
sigmaintr = para['sigmaintr']
if lgM20 == None:
    lgM1 = para['lgM1']
else:
    M1 = 20**(-1/alpha) *(10**lgM20 - 10**lgkappa * 10**lgMcut)
    lgM1 = np.log10(M1)
    
#selecting correct halo catalog
halo_fname = 'host_halos_047_M12.5.fit'

pre_title = model_name + " " + nbody + " " #pretitle for plots
kappa = 10**lgkappa
OmegaDE = 1.0 - OmegaM
H0 = 100.0*h
scale_factor = 1.0 / (1.0 + redshift)
E = (OmegaM*((1 + redshift)**3) + OmegaDE)**(0.5)

loc = "model_" + model_name + "/"

#reading in halo catalog
data, header = fitsio.read(halo_fname, header=True)
#print(header)
mass = data['M200m']
sel = (mass > Mmin)
mass = mass[sel]
x_halo = data['px'][sel]
y_halo = data['py'][sel]
z_halo = data['pz'][sel]
haloid = data['haloid'][sel]

index = np.argsort(-mass)
mass = mass[index]
x_halo = x_halo[index]
y_halo = y_halo[index]
z_halo = z_halo[index]
haloid = haloid[index]
print(f'finished reading {len(x_halo)} halos')


#reading in cluster catalog
fname = loc+'richness'+cat_name
data, header = fitsio.read(fname, header=True)
mass_cat = data['M200m']
sel = (mass_cat > Mmin)
mass_cat = mass_cat[sel]
haloid_cat = data['haloid'][sel]
Rlambda_cat = data['Rlambda'][sel]
richness_cat = data['lambda'][sel]
#print(haloid_cat)

index = np.argsort(-mass_cat)
mass_cat = mass_cat[index]
haloid_cat = haloid_cat[index]
richness_cat = richness_cat[index]

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
f_proj_myles = data[:,0]
b_lambda_myles = data[:,1]
n_bin_cl_myles = data[:,2]
mean_bin_richness_myles = data[:,3]
myles_f_proj_err = np.array([0.007, 0.021, 0.025, 0.045, 0.024, 0.025])

subvolume = total_volume / (n_parallel_x * n_parallel_y * n_parallel_z)
richness_x_axis = np.logspace(np.log10(5), np.log10(np.max(richness_cat)-1), 25)

def calc_n_greater_lambda(px_min, px_max, py_min, py_max, pz_min, pz_max):   
    #selecting halos within coordinate bounds
    sel = (px_min <= x_halo) & (x_halo <= px_max) & (py_min <= y_halo) & (y_halo <= py_max) & (pz_min <= z_halo) & (z_halo <= pz_max)
    sel_haloid = haloid[sel]
    
    #getting halos in richness catalog that are in coordinate bounds 
    sel_richness_cat = richness_cat[np.isin(haloid_cat, sel_haloid)]
    
    n_out = []
    
    for i in richness_x_axis:
        n_out.append(len(sel_richness_cat[(sel_richness_cat > i)]))
    

    return np.array(n_out)/subvolume

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
    return calc_n_greater_lambda(px_min, px_max, py_min, py_max, pz_min, pz_max)

if __name__ == '__main__':
    p = Pool(processes=8)
    results = p.map(calc_one_bin, np.arange(0,n_parallel,1))
    p.close()
    results.insert(0, richness_x_axis)
    np.savetxt(loc+'n_greater_lambda'+cat_name[:-4]+'.dat', np.transpose(results), '%g')