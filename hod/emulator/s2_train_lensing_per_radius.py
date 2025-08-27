#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

loc = '/projects/hywu/cluster_sims/cluster_finding/data/'

emu_name = sys.argv[1]
binning = sys.argv[2]
observation = sys.argv[3] #'desy1thre' # 'desy1'
rich_name = 'q180_bg_miscen' #'q180_miscen'
iz = 0
ilam = int(sys.argv[4])

alpha = 1e-6
if emu_name == 'iter1':
    alpha = 1e-3
# if emu_name == 'narrow':
#     alpha = 1e-6 
# if emu_name == 'wide':
#     alpha = 1e-6

#alpha = 1e-3 # for miscen

zid = 3+iz
train_loc = loc + f'emulator_train/{emu_name}/z0p{zid}00/{observation}_{binning}/'
plot_loc = f'../../plots/emulator_train/{emu_name}/z0p{zid}00/{observation}_{binning}/'

if os.path.isdir(plot_loc) == False:
    os.makedirs(plot_loc)

data = np.loadtxt(f'{train_loc}/parameters_all.dat')
X_all = data[:,1:]

DS_all = np.loadtxt(f'{train_loc}/DS_{binning}_bin_{ilam}_rad.dat')

ntrain, nrad = np.shape(DS_all)
print('ntrain, nrad', ntrain, nrad)

#### initial hyperparameters ####
# if alpha=0, curve goes through all points (over-training)
alpha_list = np.zeros(nrad) + alpha

length_array_ini = np.std(X_all, axis=0) 
length_scale_bounds = ([1e-4, 1e+2])

DS_recon = np.zeros((ntrain, nrad)) # reconstruct the training set

for irad in range(nrad): # train one PC at a time
    print('training rad', irad)
    y_all = DS_all[:,irad] 

    alpha = alpha_list[irad]
    kernel = np.var(y_all) * RBF(length_scale=length_array_ini, length_scale_bounds=length_scale_bounds)
    
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=9)
    gpr.fit(X_all, y_all);

    #print(f"Kernel parameters before fit:\n{kernel})")
    print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    # save the model using joblib
    joblib.dump(gpr, f'{train_loc}/DS_{binning}_bin_{ilam}_rad_{irad}_gpr_.pkl')
    # Load the model later
    # loaded_model = joblib.load('gpr_model.pkl')
    # save the kernel parameters myself
    np.savetxt(f'{train_loc}/DS_{binning}_bin_{ilam}_rad_{irad}_kernel.dat', gpr.kernel_.theta)

    DS_recon[:,irad], std_prediction = gpr.predict(X_all, return_std=True)
 
if alpha==0:
     print('recon?', np.allclose(DS_recon, DS_all)) # alpha=0 => perfect fit (passing all points)



rp = np.loadtxt(f'{train_loc}/rp_rad.dat')

# for i in range(ntrain):
#     diff = DS_recon[i] - DS_all[i]
#     plt.semilogx(rp, diff, c='gray', alpha=0.2)


#### Leave-one-simulation-out-error (LOSOE) ####

DS_recon_looe = np.zeros((ntrain, nrad))

from sklearn.gaussian_process.kernels import ConstantKernel

for ileave in range(ntrain):
    for irad in range(nrad):
        X_looe = np.delete(X_all, ileave, axis=0)
        y_looe = np.delete(DS_all[:,irad], ileave, axis=0)
        X_pred = np.array([X_all[ileave,:]])
        
        hyperpara = np.loadtxt(f'{train_loc}/DS_{binning}_bin_{ilam}_rad_{irad}_kernel.dat')#, gpr.kernel_.theta)
        hyperpara = np.exp(hyperpara)
        a = hyperpara[0]
        length_array = hyperpara[1:]
        alpha = alpha_list[irad]
        kernel = ConstantKernel(a, constant_value_bounds="fixed") * RBF(length_scale=length_array, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)#, n_restarts_optimizer=9)
        gpr.fit(X_looe, y_looe)
        
        y_pred, std = gpr.predict(X_pred, return_std=True)
        DS_recon_looe[ileave, irad] = y_pred[0]


diff = (DS_recon_looe - DS_all)
for ileave in range(ntrain):
    plt.semilogx(rp, diff[ileave], c='gray', alpha=0.2)

plt.semilogx(rp, np.median(diff, axis=0), c='C0')
plt.semilogx(rp, np.percentile(diff, 16, axis=0), c='C0')
plt.semilogx(rp, np.percentile(diff, 84, axis=0), c='C0')

plt.xlabel(r'$r_{\rm p}$')
plt.ylabel(r'$\ln \Delta\Sigma_{\rm emu} - \ln \Delta\Sigma_{\rm orig} $')
plt.title(f'LOSOE, bin {ilam}, '+ r'$\alpha$=%.e'%alpha)
#plt.ylim(-0.025, 0.025)

#### add data error bars
data_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/z0p{zid}00/model_hod000000/obs_{rich_name}_{observation}/'
rp_rad = np.loadtxt(train_loc + f'rp_rad.dat')
rp_in, DS_in = np.loadtxt(data_loc + f'DS_phys_noh_{binning}_bin_{ilam}.dat', unpack=True)
from scipy.interpolate import interp1d
DS_interp = interp1d(np.log(rp_in), np.log(DS_in))
DS_data = (np.exp(DS_interp(np.log(rp_rad))))
#rp_in, DS_in = np.loadtxt(data_loc + f'DS_phys_noh_lam_bin_{ilam}.dat', unpack=True)
cov_loc = '/users/hywu/work/cluster-lensing-cov-public/examples/abacus_summit_analytic_5k10/'
lam = [20, 30, 45, 60, 1000]
z = [0.2, 0.35, 0.5, 0.65]
rp_cov = np.loadtxt(cov_loc + f'rp_phys_noh_{z[iz]}_{z[iz+1]}_{lam[ilam]}_{lam[ilam+1]}.dat')
cov = np.loadtxt(cov_loc + f'DeltaSigma_cov_combined_phys_noh_{z[iz]}_{z[iz+1]}_{lam[ilam]}_{lam[ilam+1]}.dat')
sig = np.sqrt(np.diag(cov))
plt.semilogx(rp_rad, sig[4:]/DS_data, c='gray', ls=':', label='data error bar')
plt.semilogx(rp_rad, -sig[4:]/DS_data, c='gray', ls=':')


plt.savefig(f'{plot_loc}/emu_per_radius_err_{binning}_bin_{ilam}_alpha_{alpha:.0e}.png')
