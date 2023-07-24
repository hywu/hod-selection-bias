import numpy as np
import matplotlib.pyplot as plt
from kllr import kllr_model
import copy
import yaml
import os
import fitsio
import sys
import astropy
from astropy import cosmology
from astropy.io import fits
import astropy.units as u
import astropy.cosmology.units as cu
from scipy.interpolate import interp1d

sys.path.append('/home/andy/Documents/hod-selection-bias/repo/utils/')

from fid_hod import Ngal_S20_poisson
from fid_hod import Ngal_S20_noscatt

yml_loc = "/home/andy/Documents/hod-selection-bias/repo/utils/yml/"
yml_fname = yml_loc + 'mini_uchuu_fid_hod_scan_tophat.yml'

plt.style.use('MNRAS')

Mmin = 5e12
boxsize = 400
kllr_bins = 25

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

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

kappa = 10**lgkappa

#selecting correct halo catalog
if (redshift == 0.3):
    halo_fname = "host_halos_043.fit"
elif (redshift == 0.1):
    halo_fname = "host_halos_047.fit"

loc = "model_" + model_name + "/"

#reading in halo catalog
data, header = fitsio.read(halo_fname, header=True)
#print(header)
mass = data['M200m']
sel = (mass > Mmin)
mass = mass[sel]

#calculating richness from HOD
HOD_richness = np.array([])
for i in range(len(mass)):
    if sigmaintr < 1e-6: # poisson
        Ncen, Nsat = Ngal_S20_poisson(mass[i], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, \
                                      sigmalogM=sigmalogM) 
    HOD_richness = np.concatenate((HOD_richness, np.array([Ncen + Nsat])))

#putting filenames into cluster and member lists
dir_list = os.listdir(loc)
cluster_fnames = []
member_fnames = []
for i in dir_list:
    if "richness_" in i:
        cluster_fnames.append(i)

#sorting to ensure the ith cluster catalog matches the ith member catalog
cluster_fnames.sort()

mass_cl_cat = [mass]
richness_cl_cat = [HOD_richness]
cat_label_list = ["HOD"]
colors_cat = ['k']
plt_names = []

#setting labels and colors
for i in range(len(cluster_fnames)):
	fname = loc + cluster_fnames[i]
	data, header = fitsio.read(fname, header=True)

	mass_temp = data['M200m']

	sel = (mass_temp > Mmin)

	mass_cl_cat.append(mass_temp[sel])
	richness_cl_cat.append(data['lambda'][sel])

	legend_label = cluster_fnames[i]
	legend_label = legend_label.replace("_vel_from_part.fit","")
	legend_label = legend_label.replace("richness_","")
	plt_names.append(legend_label)

	#if statement in case you want to add different pmem models
	if 'd' in legend_label:
		legend_label = r'$d = '+legend_label[1:]+' h^{-1} {\\rm Mpc}$'
		cat_label_list.append(legend_label)

	colors_cat.append(f'C{i}')

label_richness_mean = "$\\rm \langle \lambda \\rangle$"
label_mass = "$M_{\\rm 200m} ~ [h^{-1} {\\rm Mpc}]$"

H0_astropy = 67.74 * u.km/u.s/u.Mpc
Om_astropy = 0.3089
cosmo = astropy.cosmology.FlatLambdaCDM(H0=H0_astropy, Om0=Om_astropy)

comoving_r = cosmo.comoving_distance(np.array([0.08,0.12]))
comoving_r = comoving_r.to(u.Mpc/cu.littleh, cu.with_H0(H0_astropy))
comoving_r = comoving_r.value

deg2 = 10400.288
strad = deg2 * ((np.pi/180.0)**2)
comoving_volume_myles = (strad/3) * (comoving_r[1]**3 - comoving_r[0]**3)

def costanzi_rich_mass(m):
    return 30 * ((m/3e14)**0.75)

def costanzi_std_mass(m):
    return 14.7 * ((m/3e14)**0.54)

def plot_abundance_matching():
	lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)
	kllr_results = []

	for i in range(len(richness_cl_cat)):
	    temp = lm.fit(np.log(mass_cl_cat[i]), richness_cl_cat[i], bins=kllr_bins)
	    kllr_results.append(np.exp(temp[0]))
	    kllr_results.append(np.mean(temp[1], axis = 0))
	    kllr_results.append(np.mean(temp[4], axis = 0))
	    kllr_results.append(np.std(temp[4], axis = 0))
	    
	kllr_results = np.array(kllr_results)
	kllr_results = np.reshape(kllr_results, (len(richness_cl_cat), 4, kllr_bins))

	SDSS_richness = np.array(np.loadtxt('SDSS_richness.dat'))

	#making cdf for SDSS richness
	n_SDSS = []
	for i in range(len(SDSS_richness)):
	    n_SDSS.append(len(SDSS_richness[ (SDSS_richness > SDSS_richness[i]) ]))
	n_SDSS = np.array(n_SDSS) / comoving_volume_myles

	#inteprolating cdf of SDSS richness
	reverse_richness_CDF_SDSS = interp1d(n_SDSS, SDSS_richness)

	matched_lambda_SDSS = []
	matched_N_cyl = []
	matched_lambda_SDSS_all = []

	slope_lambda_SDSS_n_cyl = np.zeros(len(cluster_fnames))
	intercept_lambda_SDSS_n_cyl = np.zeros(len(cluster_fnames))

	for i in range(len(cluster_fnames)):
	    #making cdf of N_cyl
	    n_cyl = []
	    for j in range(len(richness_cl_cat[i+1])):
	        n_cyl.append( len(richness_cl_cat[i+1][ (richness_cl_cat[i+1] > richness_cl_cat[i+1][j]) ]) )
	    n_cyl = np.array(n_cyl) / (boxsize**3)
	    
	    #selecting within interpolation range
	    sel = (np.min(n_SDSS) <= n_cyl) & (n_cyl <= np.max(n_SDSS))
	    matched_lambda_SDSS.append(reverse_richness_CDF_SDSS(n_cyl[sel]))
	    matched_N_cyl.append(richness_cl_cat[i+1][sel])
	    
	    #sorting so plots look nice
	    sel = np.argsort(matched_N_cyl[i])
	    matched_lambda_SDSS[i] = matched_lambda_SDSS[i][sel]
	    matched_N_cyl[i] = matched_N_cyl[i][sel]
	    
	    #linear fit between log(SDSS_richness) and log(N_cyl)
	    slope_lambda_SDSS_n_cyl[i], intercept_lambda_SDSS_n_cyl[i] = np.polyfit(np.log10(matched_N_cyl[i]), \
	                                                                 np.log10(matched_lambda_SDSS[i]), 1)
	    
	    intercept_lambda_SDSS_n_cyl[i] = 10**intercept_lambda_SDSS_n_cyl[i]
	    
	    matched_lambda_SDSS_all.append(intercept_lambda_SDSS_n_cyl[i]*(richness_cl_cat[i+1]**slope_lambda_SDSS_n_cyl[i]))


	lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)
	kllr_results_ab = []

	for i in range(len(cluster_fnames)):
	    temp = lm.fit(np.log(mass_cl_cat[i+1]), matched_lambda_SDSS_all[i], bins=kllr_bins)
	    kllr_results_ab.append(np.exp(temp[0]))
	    kllr_results_ab.append(np.mean(temp[1], axis = 0))
	    kllr_results_ab.append(np.mean(temp[4], axis = 0))
	    kllr_results_ab.append(np.std(temp[4], axis = 0))
	    
	kllr_results_ab = np.array(kllr_results_ab)
	kllr_results_ab = np.reshape(kllr_results_ab, (len(cluster_fnames), 4, kllr_bins))

	plt.plot(kllr_results[0][0], kllr_results[0][1], label=cat_label_list[0], c=colors_cat[0])

	for i in range(len(cluster_fnames)):
	    plt.plot(kllr_results_ab[i][0], kllr_results_ab[i][1], c=colors_cat[i+1], ls='dashed')
	    plt.plot(kllr_results[i+1][0], kllr_results[i+1][1], c=colors_cat[i+1], label=cat_label_list[i+1])

	plt.plot(kllr_results[0][0], kllr_results[0][1], label=cat_label_list[0], c=colors_cat[0])
	plt.plot([1e13],[100], ls='dashed', label='abundance matched', c='k')
	plt.xlabel(label_mass)
	plt.ylabel(label_richness_mean)
	plt.loglog()

	fname_out = 'richness_mass'
	for i in range(len(plt_names)):
		fname_out+=plt_names[i]
	plt.savefig(fname_out+'.png')

	plt.show()

def plot_lensing():
	rp_SDSS, DS_SDSS = np.loadtxt(loc + 'lensing/deltaSigma_pbrcorrected.dat', unpack=True) 
	cov_SDSS = np.load(loc + 'lensing/deltaSigma_pbrcorrected_jk_cov_t.dat.npy')
	dDS_SDSS = np.sqrt(np.diag(cov_SDSS))
	sel = rp_SDSS > 0.2

	dir_list = os.listdir(loc + 'lensing/')
	fname_cyl_lensing = []

	for i in range(len(dir_list)):
	    if "DS_abun_bin_0_" in dir_list[i]:
	        fname_cyl_lensing.append(dir_list[i])

	label_lensing = copy.deepcopy(fname_cyl_lensing)

	for i in range(len(label_lensing)):
	    label_lensing[i] = label_lensing[i].replace("DS_abun_bin_0_","")
	    label_lensing[i] = label_lensing[i].replace(".dat","")
	    label_lensing[i] = label_lensing[i].replace("d30", "$d = 30 ~h^{-1} {\\rm Mpc}$")
	    label_lensing[i] = label_lensing[i].replace("d60", "$d = 60 ~h^{-1} {\\rm Mpc}$")
	    label_lensing[i] = label_lensing[i].replace("d90", "$d = 90 ~h^{-1} {\\rm Mpc}$")
	    label_lensing[i] = label_lensing[i].replace("d120", "$d = 120 ~h^{-1} {\\rm Mpc}$")
	#plt.rcParams['text.usetex'] = True
	#reading in d90
	#rp_d90, DS_d90, x_d90 = np.loadtxt(loc+'lensing/DS_abun_bin_0.dat',unpack=True)

	plt.errorbar(rp_SDSS[sel], (rp_SDSS*DS_SDSS)[sel], (rp_SDSS*dDS_SDSS)[sel], label='SDSS', \
	             capsize=14, c='k', ls='',marker='o', mec='k')

	for i in range(len(fname_cyl_lensing)):
	    rp, DS, x = np.loadtxt(loc + 'lensing/' + fname_cyl_lensing[i], unpack=True)
	    
	    plt.plot(rp, rp*DS, label=label_lensing[i])

	#plt.plot(rp_d90,rp_d90*DS_d90, label=r'$\rm d = 90 ~h^{-1} Mpc$', c='C0')
	plt.xscale('log')
	plt.legend(framealpha=0, bbox_to_anchor=(1.05, 1.0), loc='upper left')
	plt.xlabel(r'$r_p~[h^{-1} {\rm Mpc}]$')
	plt.ylabel(r'$r_p\Delta\Sigma~[{\rm Mpc ~M_\odot pc^{-2}}]$')
	plt.savefig('SDSS_lensing.png')
	plt.show()

if __name__=='__main__':
	plot_lensing()
	plot_abundance_matching()
