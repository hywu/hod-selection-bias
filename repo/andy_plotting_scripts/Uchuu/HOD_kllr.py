import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import fitsio
from kllr import kllr_model
from multiprocessing import Pool

sys.path.append('/home/andy/Documents/hod-selection-bias/repo/utils/')
from fid_hod import Ngal_S20_poisson
from fid_hod import Ngal_S20_noscatt

yml_loc = "/home/andy/Documents/hod-selection-bias/repo/utils/yml/"
yml_fname_list = ['uchuu_fid_hod.yml']
yml_fname = yml_loc + yml_fname_list[0]

Mmin=5e12

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
kappa = 10**lgkappa
lgMcut = para['lgMcut']
sigmalogM = para['sigmalogM']
sigmaintr = para['sigmaintr']
if lgM20 == None:
    lgM1 = para['lgM1']
else:
    M1 = 20**(-1/alpha) *(10**lgM20 - 10**lgkappa * 10**lgMcut)
    lgM1 = np.log10(M1)
loc = "model_" + model_name + "/"

processors = 8
z_thickness = 2000.0/processors

def calc_one_bin(i):
    z_lower = z_thickness*i
    z_upper = z_thickness*(i+1)
    sel = (z_lower < z_halo) & (z_halo < z_upper)
    
    mass_sel = mass[sel]
    HOD_richness = np.array([])
    for i in range(len(mass_sel)):
        if sigmaintr < 1e-6: # poisson
            Ncen, Nsat = Ngal_S20_poisson(mass_sel[i], alpha=alpha, lgM1=lgM1, kappa=kappa, lgMcut=lgMcut, \
                                          sigmalogM=sigmalogM) 
        HOD_richness = np.concatenate((HOD_richness, np.array([Ncen + Nsat])))
    return mass_sel, HOD_richness

if __name__=='__main__':
    p = Pool(processes=processors)
    results = p.map(calc_one_bin, np.arange(0,processors,1))
    p.close()

    mass_master = np.array([])
    richness_master = np.array([])

    for i in range(processors):
        mass_master = np.concatenate((mass_master, results[i][0]))
        richness_master = np.concatenate((richness_master, results[i][1]))

    kllr_results = []

    lm = kllr_model(kernel_type='gaussian', kernel_width = 0.2)
    temp = lm.fit(np.log(mass_master), richness_master, bins=25)

    kllr_results.append(np.exp(temp[0]))
    kllr_results.append(np.mean(temp[1], axis=0))
    kllr_results.append(np.mean(temp[4], axis=0))
    kllr_results.append(np.std(temp[4], axis=0))

    np.savetxt(loc+'kllr_HOD.dat', np.transpose(kllr_results), '%g', header='mass, richness, richness_err, richness_err_err')

