import fitsio
import numpy as np
import scipy
import matplotlib.pyplot as plt
import copy
import astropy
import yaml
import os
from astropy import cosmology
from astropy.io import fits
import astropy.units as u
import astropy.cosmology.units as cu
from scipy import spatial
from scipy import special
from scipy.optimize import curve_fit
from scipy.optimize import Bounds
from scipy.optimize import minimize
from multiprocessing import Pool

#is it good practice to unappend the path?
import sys
sys.path.append('/home/andy/Documents/hod-selection-bias/repo/utils/')

yml_fname_list = ['uchuu_fid_hod.yml']
yml_fname = '/home/andy/Documents/hod-selection-bias/repo/utils/yml/'+yml_fname_list[0]

cat_name = sys.argv[1] #quad200
cat_name = '_' + cat_name + '.fit'


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

'''
#reading in galaxy catalog
data, header = fitsio.read(loc+"gals.fit", header=True)
#print(header)
x_gal = data['px']
y_gal = data['py']
z_gal = data['pz']
print(f'finished reading {len(x_gal)} galaxies')
'''

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
Rlambda_cat = Rlambda_cat[index]
richness_cat = richness_cat[index]

#reading in member catalog
fname = loc+'members'+cat_name
data, header = fitsio.read(fname, header=True)
print(header)
haloid_mem = data['haloid']
pmem_mem = data['pmem']
dz_gal_mem = data['dz_gal']

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
volume_ratio = subvolume / comoving_volume_myles
abundance_match_n = volume_ratio * n_bin_cl_myles
abundance_match_n = np.rint(abundance_match_n)

myles_bin_edges = np.array([5, 20, 27.9, 37.6, 50.3, 69.3, 140])
n_bins_myles = 6

#inital guess for minimizer
x0_f_cl = np.array([0.6, 0.74, 0.79, 0.82, 0.83, 0.92])
x0_sigma_cl = np.array([4.0,6.0,7.0,8.0,9.0,11.0])

x0_quad = np.zeros(13)
x0_quad[0:n_bins_myles] = x0_f_cl
x0_quad[n_bins_myles: 2*n_bins_myles] = x0_sigma_cl
x0_quad[-1] = 210

#bounds for minimizer
bounds_lower_quad = np.zeros(13) + 0.5
bounds_lower_quad[n_bins_myles:2*n_bins_myles] = 1
bounds_lower_quad[-1] = 100
bounds_upper_quad = np.ones(13)
bounds_upper_quad[n_bins_myles:2*n_bins_myles] = 13
bounds_upper_quad[-1] = 1e3
bounds_quad = Bounds(bounds_lower_quad, bounds_upper_quad)


def norm(x, sigma):
    return np.exp(-0.5 * x**2 / sigma**2)/(sigma*np.sqrt(2*np.pi))

def lnlikelihood_single_quad_guass(delta_chi, f_cl, sigma_cl, q):
    p_out = f_cl*norm(delta_chi, sigma_cl)
    sel = (abs(delta_chi) < q)
    
    p_out[sel] += 3*(1.0-f_cl)*(1-(delta_chi[sel]/q)**2)/(4*q)
    
    sel = (p_out < 1e-20)
    p_out[sel] = 1e-20
    return np.log(p_out)

def calc_dbl_gauss_param(px_min, px_max, py_min, py_max, pz_min, pz_max):   
    #selecting halos within coordinate bounds
    sel = (px_min <= x_halo) & (x_halo <= px_max) & (py_min <= y_halo) & (y_halo <= py_max) & (pz_min <= z_halo) & (z_halo <= pz_max)
    sel_haloid = haloid[sel]
    
    #getting halos in richness catalog that are in coordinate bounds 
    sel_richness_cat = richness_cat[np.isin(haloid_cat, sel_haloid)]
    
    richness_sort = np.argsort(sel_richness_cat)
    sort_haloid = sel_haloid[richness_sort]
    
    sort_haloid = sort_haloid[-int(np.sum(abundance_match_n)):]
    
    binid = np.array([])
    dz_gal = np.array([])
    pmem = np.array([])
    
    for i in range(n_bins_myles):
        index_slice = int(abundance_match_n[i])
        
        haloid_temp = sort_haloid[:index_slice]
        
        sort_haloid = sort_haloid[index_slice:]

        sel_members = np.isin(haloid_mem, haloid_temp)
        
        delz = dz_gal_mem[sel_members]
        pmem_temp = pmem_mem[sel_members]
        
        binid = np.concatenate((binid, np.zeros(len(delz))+i))
        dz_gal = np.concatenate((dz_gal, delz))
        pmem = np.concatenate((pmem, pmem_temp))
        
    def negative_lnlikelihood_total(theta):
        tot = 0
        
        f_cl_l = theta[0:n_bins_myles]
        sigma_cl_l = theta[n_bins_myles:2*n_bins_myles]
        q_l = theta[-1]
        
        for i in range(n_bins_myles):
            sel = (binid == i) & (dz_gal != 0)
            
            tot += -np.sum(pmem[sel]*lnlikelihood_single_quad_guass(dz_gal[sel], f_cl_l[i], sigma_cl_l[i], \
                                                             q_l))
            
        if not np.isfinite(tot):
            return -np.inf
        
        return tot
    
    res = minimize(negative_lnlikelihood_total, x0_quad, bounds=bounds_quad,\
                  method='Nelder-Mead', options={'maxiter':100000}).x
#     f_cl = res[0:n_bins_myles]
#     sigma_cl = res[n_bins_myles:2*n_bins_myles]
#     sigma_proj = res[-1]
#     for i in range(n_bins_myles):
#         sel = (binid == i) & (dz_gal != 0)
#         plot_input = np.linspace(-np.max(abs(dz_gal[sel])), np.max(abs(dz_gal[sel])), 1000)
#         plt.hist(dz_gal[sel], density=True, bins=250, weights=pmem[sel])
#         plt.plot(plot_input, dbl_gauss_pdf(plot_input, f_cl[i], sigma_cl[i], sigma_proj))
#         plt.yscale('log')
#         plt.show()
    return res
    
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
    
    return calc_dbl_gauss_param(px_min, px_max, py_min, py_max, pz_min, pz_max)


if __name__=='__main__':
    
    p = Pool(processes=1)
    theta_list = p.map(calc_one_bin, np.arange(0,n_parallel,1))
    p.close()

    np.savetxt(f'model_{model_name}/params{cat_name[:-4]}.dat', np.transpose(theta_list), '%g', header='Column is param for 1 subvolume')