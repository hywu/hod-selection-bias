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
binning = sys.argv[2]  #'AB' # 'lam' # 'abun'
iz = 0

zid = 3+iz
train_loc = loc + f'emulator_train/{emu_name}/z0p{zid}00/{binning}'
plot_loc = f'../../plots/emulator_train/{emu_name}/z0p{zid}00/{binning}'

if os.path.isdir(plot_loc) == False:
    os.makedirs(plot_loc)

alpha = 1e-5 
if emu_name == 'iter1':
    alpha = 1e-3
# if emu_name == 'narrow':
#     alpha = 1e-5 
# if emu_name == 'wide':
#     alpha = 1e-5


# 1e-3 => too noisy
# 1e-4 => okay
# 1e-5 => best
# 1e-6 => too noisy

#data = np.loadtxt(f'{train_loc}/cosmologies_all.dat')
data = np.loadtxt(f'{train_loc}/parameters_all.dat')
X_all = data[:,1:]
abun_all = np.loadtxt(f'{train_loc}/abundance.dat')

ntrain, nbin = np.shape(abun_all)
print('ntrain, nbin', ntrain, nbin)

#### initial hyperparameters ####
# if alpha=0, curve goes through all points (over-training)
alpha_list = np.zeros(nbin) + alpha

length_array_ini = np.std(X_all, axis=0) 
length_scale_bounds = ([1e-4, 1e+2])

abun_recon = np.zeros((ntrain, nbin)) # reconstruct the training set

for ilam in range(nbin): # train one bin at a time
    print('training bin', ilam)
    y_all = abun_all[:,ilam] 

    alpha = alpha_list[ilam]
    kernel = np.var(y_all) * RBF(length_scale=length_array_ini, length_scale_bounds=length_scale_bounds)
    
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=9)
    gpr.fit(X_all, y_all);

    #print(f"Kernel parameters before fit:\n{kernel})")
    print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    # save the model using joblib
    joblib.dump(gpr, f'{train_loc}/abundance_bin_{ilam}_gpr_.pkl')
    # Load the model later
    # loaded_model = joblib.load('gpr_model.pkl')
    # save the kernel parameters myself
    np.savetxt(f'{train_loc}/abundance_bin_{ilam}_kernel.dat', gpr.kernel_.theta)

    abun_recon[:,ilam], std_prediction = gpr.predict(X_all, return_std=True)
 
if alpha==0:
     print('recon?', np.allclose(abun_recon, abun_all)) # alpha=0 => perfect fit (passing all points)



#### Leave-one-simulation-out-error (LOSOE) ####

abun_recon_looe = np.zeros((ntrain, nbin))

from sklearn.gaussian_process.kernels import ConstantKernel

for ileave in range(ntrain):
    for ilam in range(nbin):
        X_looe = np.delete(X_all, ileave, axis=0)
        y_looe = np.delete(abun_all[:,ilam], ileave, axis=0)
        X_pred = np.array([X_all[ileave,:]])
        
        hyperpara = np.loadtxt(f'{train_loc}/abundance_bin_{ilam}_kernel.dat')#, gpr.kernel_.theta)
        hyperpara = np.exp(hyperpara)
        a = hyperpara[0]
        length_array = hyperpara[1:]
        alpha = alpha_list[ilam]
        kernel = ConstantKernel(a, constant_value_bounds="fixed") * RBF(length_scale=length_array, length_scale_bounds="fixed")
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)#, n_restarts_optimizer=9)
        gpr.fit(X_looe, y_looe)
        
        y_pred, std = gpr.predict(X_pred, return_std=True)
        abun_recon_looe[ileave, ilam] = y_pred[0]


diff = (abun_recon_looe - abun_all)
#print(diff)

for ileave in range(ntrain):
    plt.plot(diff[ileave], c='gray', alpha=0.2)

plt.plot(np.median(diff, axis=0), c='C0')
plt.plot(np.percentile(diff, 16, axis=0), c='C0')
plt.plot(np.percentile(diff, 84, axis=0), c='C0')

plt.xlabel('richness bin')
plt.ylabel(r'$\ln N_{\rm emu} - \ln N_{\rm orig} $')
plt.title(f'LOSOE, '+ r'$\alpha$=%.e'%alpha)
#plt.ylim(-0.025, 0.025)

#### add data error bars
data_loc = f'/projects/hywu/cluster_sims/cluster_finding/data/emulator_data/base_c000_ph000/z0p{zid}00/model_hod000000/obs_q180_desy1/'
x, x, NC_data = np.loadtxt(data_loc+'abundance.dat',unpack=True)
cov_loc = '/users/hywu/work/cluster-lensing-cov-public/examples/abacus_summit_analytic/'
#cov_NC = np.diag(NC_data)
lam = [20, 30, 45, 60, 1000]
z = [0.2, 0.35, 0.5, 0.65]
cov_NC = []
for ilam in range(4):
    counts, sv, bias, lnM_mean = np.loadtxt(cov_loc + f'counts_{z[iz]}_{z[iz+1]}_{lam[ilam]}_{lam[ilam+1]}.dat')
    cov_NC.append(counts + sv)
cov_NC = np.diag(cov_NC)
#cov_NC_inv = linalg.inv(cov_NC)
sig_NC = np.sqrt(np.diag(cov_NC))
plt.plot(sig_NC/NC_data, c='gray', ls=':')
plt.plot(-sig_NC/NC_data, c='gray', ls=':')

plt.savefig(f'{plot_loc}/emu_abundance_err_alpha_{alpha:.0e}.png')
