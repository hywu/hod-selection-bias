import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import fitsio
import scipy
from scipy.optimize import curve_fit

sys.path.append('/home/andy/Documents/hod-selection-bias/repo/utils/')
from fid_hod import Ngal_S20_poisson
from fid_hod import Ngal_S20_noscatt

yml_loc = "/home/andy/Documents/hod-selection-bias/repo/utils/yml/"
yml_fname_list = ['uchuu_fid_hod.yml']
yml_fname = yml_loc + yml_fname_list[0]

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

dir_list = os.listdir(loc)

param_fnames = []
param_list = []
label_list = []
plt_name = []

label_lambda_z = []
fname_lambda_z = []

fname_n = []
label_n = []

fname_kllr = []
label_kllr = []

fname_out_redshift_scan = 'redshift_scan'

#formatting legend label in plots and outputted plot names
for i in dir_list:
    if 'params' in i:
        param_fnames.append(i)
        param_list.append(np.loadtxt(loc+i))
        if 'gauss' in i:
            label_list.append(r'$\sigma = ' + i[12:-4] + '~h^{-1} {\\rm Mpc}$')
            plt_name.append((i[6:])[:-4])
        elif 'quad' in i:
            label_list.append(r'$q = ' + i[11:-4] + '~h^{-1} {\\rm Mpc}$')
            plt_name.append((i[6:])[:-4])
        elif 'tophat' in i:
            label_list.append(r'$d = ' + i[13:-4] + '~h^{-1} {\\rm Mpc}$')
            plt_name.append((i[6:])[:-4])
            
    elif ('w_' in i) & ('_scan.dat' in i):
        fname_lambda_z.append(i)
        if 'gauss' in i:
            label_lambda_z.append(r'$\sigma = '+i[7:-9]+'~ h^{-1} {\\rm Mpc}$')
            fname_out_redshift_scan += '_'+i[2:-9]
        elif 'quad' in i:
            label_lambda_z.append(r'$q = '+i[6:-9]+'~ h^{-1} {\\rm Mpc}$')
            fname_out_redshift_scan += '_'+i[2:-9]
    elif ('n_greater_lambda_' in i) & ('.dat' in i):
        fname_n.append(i)
        if 'gauss' in i:
            label_n.append(r'$\sigma = '+i[22:-4]+'~ h^{-1} {\\rm Mpc}$')
        elif 'quad' in i:
            label_n.append(r'$q = '+i[21:-4]+'~ h^{-1} {\\rm Mpc}$')
        elif 'tophat' in i:
            label_n.append(r'$d = '+i[23:-4]+'~ h^{-1} {\\rm Mpc}$')
            
    elif ('kllr_' in i) & ('.dat' in i):
        fname_kllr.append(i)
        if 'gauss' in i:
            label_kllr.append(r'$\sigma ='+i[10:-4]+'~ h^{-1} {\\rm Mpc}$')
        elif 'quad' in i:
            label_kllr.append(r'$q = '+i[9:-4]+'~ h^{-1} {\\rm Mpc}$')
        elif 'tophat' in i:
            label_kllr.append(r'$d = '+i[11:-4]+'~ h^{-1} {\\rm Mpc}$')
            
            
fname_out_redshift_scan += '.png'

fname_fproj_richness = 'fproj_richness'
fname_sigmacl_richness = 'sigmacl_richness'
fname_sigmaproj_richess = 'sigmaproj_richness'

for i in range(len(plt_name)):
    fname_fproj_richness += plt_name[i]
    fname_sigmacl_richness += plt_name[i]
    fname_sigmaproj_richess += plt_name[i]

fname_fproj_richness += '.png'
fname_sigmacl_richness += '.png'
fname_sigmaproj_richess += '.png'

plt.style.use('MNRAS')  
capsize = 10

label_f_proj = "$f_{\\rm proj}$"
label_richness_mean = "$\langle \lambda \\rangle$"
label_mass = "$M_{\\rm 200m} ~ [h^{-1} {\\rm Mpc}]$"
label_sigma_cl = "$\sigma_{\\rm cl} ~ [h^{-1} {\\rm Mpc}]$"
label_sigma_proj = "$\sigma_{\\rm proj} ~ [{\\rm km/s}]$"

data = np.loadtxt('myles_data.dat')
f_proj_myles = data[:,0]
b_lambda_myles = data[:,1]
b_lambda_err_myles = np.array([0.006, 0.020, 0.024, 0.042, 0.022, 0.030])
n_bin_cl_myles = data[:,2]
mean_bin_richness_myles = data[:,3]
f_proj_err_myles = np.array([0.007, 0.021, 0.025, 0.045, 0.024, 0.025])
sigma_proj_myles = 8689.0
sigma_proj_err_myles = 1074.0

sigma_cl_myles = np.array([379, 503, 614, 634, 770, 1060], dtype=float)
sigma_cl_err_myles = np.array([6,20,30,40,40,139], dtype=float)

bin_edges_myles = np.array([5, 20, 27.9, 37.6, 50.3, 69.3, 140])
n_bins_myles = len(bin_edges_myles)-1

z = 0.1
Om = 0.3089
Ez = np.sqrt(Om * (1 + z)**3 + (1 - Om))
c = 3e5

def unit_conversion(v=None, dz=None, dchi=None):
    if v:
        dz_out = v / c * (1+z)
        dchi_out = dz_out * 3000. / Ez
        v_out = v * 1.
    if dz:
        dchi_out = dz * 3000. / Ez
        v_out = c * dz / (1+z)
        dz_out = dz * 1.
    if dchi:
        v_out = c* Ez /3000. * dchi / (1+z)
        dz_out = dchi * Ez / 3000.
        dchi_out = dchi * 1.
    #print('v=%.3g km/s, dz=%.3g, dchi=%.3g Mpc/h'%(v_out, dz_out, dchi_out))
    return v_out, dz_out, dchi_out

def cylinder_conversion(d=None, q=None, sigma=None):
    if d:
        q_out = 15./8. * d
        sigma_out = 2./np.sqrt(np.pi) * d
        d_out = d * 1.
    if q:
        sigma_out = 2./np.sqrt(np.pi) * 8./ 15. * q
        d_out = 8./15. * q
        q_out = q * 1.
    if sigma:
        d_out = np.sqrt(np.pi)/2 * sigma
        q_out = np.sqrt(np.pi)/2 * 15./8. * sigma
        sigma_out = sigma * 1.
    #print('d=%.3g, q=%.3g, sigma=%.3g Mpc/h'%(d_out, q_out, sigma_out))
    return d_out, q_out, sigma_out   

def dchi2dz(dchi):
    return dchi * Ez / 3000.

def dz2dchi(dz):
    return dz * 3000. / Ez

def dchi2v(dchi):
    return c* Ez /3000. * dchi / (1+z)

def v2dchi(v):
    dz_out = v / c * (1+z)
    return dz_out * 3000. / Ez

sigma_proj_myles = unit_conversion(v=sigma_proj_myles)[2]
sigma_proj_err_myles = unit_conversion(v=sigma_proj_err_myles)[2]

for i in range(len(sigma_cl_myles)):
    sigma_cl_myles[i] = unit_conversion(v=sigma_cl_myles[i])[2]
    sigma_cl_err_myles[i] = unit_conversion(v=sigma_cl_err_myles[i])[2]
    
def norm(x, sigma):
    return np.exp(-0.5 * x**2 / sigma**2)/(sigma*np.sqrt(2*np.pi))
    
def dbl_gauss_proj(delta_chi, f_cl, sigma_proj):
    return (1.0 - f_cl)*norm(delta_chi, sigma_proj)

def quad_proj(delta_chi, f_cl, q):
    p_out = np.zeros(np.shape(delta_chi))
    sel = (abs(delta_chi) < q)
    p_out[sel] = 3*(1.0-f_cl)*(1-(delta_chi[sel]/q)**2)/(4*q)
    return p_out

lambda_p = 33.336
z_p = 0.171
sigma_p = v2dchi(618.1)
alpha = 0.435
beta = 0.54

def rozo2015(richness):
    return sigma_p*(((1+0.1)/(1+z_p))**beta)*((richness/lambda_p)**alpha)

def murata2018(m, sigma0, q):
    return sigma0 + q * np.log(m/(3e14))

def plot_projection_component():
    fig = plt.figure(figsize = (plt.rcParams.get('figure.figsize')[0]*3, \
                                    plt.rcParams.get('figure.figsize')[1]*2))
    gs = fig.add_gridspec(2, 3, hspace=0, wspace=0)
    ax = gs.subplots(sharex=True, sharey=True)

    for catalog in range(len(param_fnames)):
        mean_param = np.mean(param_list[catalog], axis=1)
        f_cl = mean_param[0:n_bins_myles]
        
        if not 'tophat' in param_fnames[catalog]:
            param_bg = mean_param[-1]
            
        for j in range(n_bins_myles):
            if (j != 0):
                label = f"{bin_edges_myles[j]} $< \lambda\leq$ {bin_edges_myles[j+1]}"
            else:
                label = f"{bin_edges_myles[j]} $\leq\lambda\leq$ {bin_edges_myles[j+1]}"
                
            plt_i = j // 3
            plt_j = j % 3
            if (j != 0):
                label = f"{bin_edges_myles[j]} $< \lambda\leq$ {bin_edges_myles[j+1]}"
            else:
                label = f"{bin_edges_myles[j]} $\leq\lambda\leq$ {bin_edges_myles[j+1]}"

            dbl_gauss_input = np.linspace(-300, 300, 1000)
            if 'gauss' in param_fnames[catalog]:
                ax[plt_i, plt_j].plot(dbl_gauss_input, dbl_gauss_proj(dbl_gauss_input, f_cl[j], param_bg), \
                                     label=label_list[catalog])
            elif 'quad' in param_fnames[catalog]:
                ax[plt_i, plt_j].plot(dbl_gauss_input, quad_proj(dbl_gauss_input, f_cl[j], param_bg), \
                                     label=label_list[catalog])
            ax[plt_i,plt_j].annotate(label, xy=(0.6,0.9), xycoords='axes fraction')
                
    for j in range(n_bins_myles):
        plt_i = j // 3
        plt_j = j % 3
        dbl_gauss_input = np.linspace(-300, 300, 1000)
        ax[plt_i, plt_j].plot(dbl_gauss_input, dbl_gauss_proj(dbl_gauss_input, 1.0-f_proj_myles[j],\
                              sigma_proj_myles), label = 'SDSS', c='k')
        ax[plt_i, plt_j].set_yscale('log')
        ax[plt_i, plt_j].set_ylim(bottom=5e-5, top=3e-3)
        
    ax[-1,-1].legend(framealpha=0, loc='upper left')
    # fig.suptitle(f"{cat_label_arr[i]}")
    fig.supxlabel(r'$\Delta \chi [h^{-1} {\rm Mpc}]$')
    fig.supylabel('PDF')
    fname_out = 'proj_comp'
    for i in range(len(plt_name)):
        fname_out += plt_name[i]
    fname_out += '.png'
    fig.savefig(loc+fname_out)

def plot_f_proj():
    plt.errorbar(mean_bin_richness_myles, f_proj_myles, yerr=f_proj_err_myles, label='SDSS', c='k', capsize=capsize, \
                marker='o')
    for i in range(len(param_fnames)):
        mean_param = np.mean(param_list[i], axis=1)
        std_param = np.std(param_list[i], axis=1)
        plt.errorbar(mean_bin_richness_myles+i+1, 1-mean_param[:n_bins_myles], yerr=std_param[:n_bins_myles],\
                    label=label_list[i], capsize=capsize)
        
    plt.xlabel(label_richness_mean)
    plt.ylabel(label_f_proj)
    plt.legend(framealpha=0)
    plt.savefig(loc+fname_fproj_richness)
    plt.show()

def plot_sigma_cl():
    fig, ax = plt.subplots()
    ax.errorbar(mean_bin_richness_myles, sigma_cl_myles, yerr=sigma_cl_err_myles, label='SDSS', c='k', \
                 capsize=capsize, marker='o')
    for i in range(len(param_fnames)):
        mean_param = np.mean(param_list[i], axis=1)
        std_param = np.std(param_list[i], axis=1)
        ax.errorbar(mean_bin_richness_myles+i+1, mean_param[n_bins_myles:2*n_bins_myles], \
                     yerr=std_param[n_bins_myles:2*n_bins_myles],\
                    label=label_list[i], capsize=capsize)
    ax.plot(mean_bin_richness_myles, rozo2015(mean_bin_richness_myles), label='Rozo et al. 2015')
    ax.set_xlabel(label_richness_mean)
    ax.set_ylabel(label_sigma_cl)
    ax.legend(framealpha=0)
    ax.tick_params(axis='y', which='both', left=True, right=False)
    secax = ax.secondary_yaxis('right', functions=(dchi2v, v2dchi))
    secax.set_ylabel(r'$v ~[{\rm km ~ s^{-1}}]$')
    ax.set_ylim(top=14.5)
    fig.savefig(loc+fname_sigmacl_richness)

def plot_redshift_scan():
    delta_chi = np.arange(-200,201,5)

    fig, ax = plt.subplots()
    for i in range(len(fname_lambda_z)):
        data_in = np.loadtxt(loc+fname_lambda_z[i])
        mean_lambda_z = np.mean(data_in, axis=1)
        std_lambda_z = np.std(data_in, axis=1)
        
        mean_lambda_z = np.concatenate((np.flip(mean_lambda_z[1:]), mean_lambda_z))
        std_lambda_z = np.concatenate((np.flip(std_lambda_z[1:]), std_lambda_z))
        
        ax.errorbar(delta_chi, mean_lambda_z, yerr=std_lambda_z, label=label_lambda_z[i])
    ax.set_xlabel(r'$\Delta \chi ~ [h^{-1} {\rm Mpc}]$')
    ax.set_ylabel(r'$\lambda(\Delta \chi)$')
    ax.legend(framealpha=0)
    ax.tick_params(axis='x', which='both', top=False, bottom=True)
    secax = ax.secondary_xaxis('top', functions=(dchi2dz, dz2dchi))
    secax.set_xlabel(r'$\Delta z$')
    dist, lam = np.loadtxt('sdss_lambda_z_avg.dat', unpack=True)
    plt.plot(dist[dist<300], lam[dist<300], label='SDSS', c='k')
    fig.savefig(loc+fname_out_redshift_scan)
    plt.show()

def plot_n_greater_lambda():
    data_in = np.loadtxt('n_greater_lambda_SDSS.dat')
    plt.plot(data_in[:,0], data_in[:,1], label='SDSS', c='k')

    for i in range(len(fname_n)):
        data_in = np.loadtxt(loc+fname_n[i])
        richness_x_axis = data_in[:,0]
        n_greater_lambda = np.mean(data_in[:,1:], axis=1)
        n_greater_lambda_err = np.std(data_in[:,1:], axis=1)
        plt.plot(richness_x_axis, n_greater_lambda, label=label_n[i])
        #plt.errorbar(richness_x_axis, n_greater_lambda, yerr=n_greater_lambda_err, label=label_n[i])
    #     plt.fill_between(richness_x_axis, n_greater_lambda-n_greater_lambda_err, \
    #                      n_greater_lambda+n_greater_lambda_err, label=label_n[i], alpha=0.2)
    plt.loglog()
    plt.ylim(bottom=1e-7)
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$n(>\lambda)~[h^3 {\rm Mpc^{-3}}]$')
    plt.legend(framealpha=0)
    plt.xlim(left=5, right=1.5e2)

    fname_out = 'n_greater_lam'
    for i in range(len(fname_n)):
        fname_out += '_'+fname_n[i]
    fname_out += '.png'
    plt.savefig(loc+fname_out)
    plt.show()

def plot_richness_mass():
    for i in range(len(fname_kllr)):
        data_in = np.loadtxt(loc+fname_kllr[i])
        plt.plot(data_in[:,0], data_in[:,1], label=label_kllr[i])
        
    data_in = np.loadtxt('kllr_HOD.dat')
    plt.plot(data_in[:,0], data_in[:,1], label='HOD', c='k')
    plt.loglog()
    plt.legend(framealpha=0)
    plt.xlabel(label_mass)
    plt.ylabel(label_richness_mean)
    plt.savefig(loc+'richness_mass.png')
    plt.show()

def plot_sigma_lambda_ov_lambda():
    for i in range(len(fname_kllr)):
        data_in = np.loadtxt(loc+fname_kllr[i])
        plt.plot(data_in[:,0], data_in[:,2]/data_in[:,1], label=label_kllr[i])
        
    data_in = np.loadtxt('kllr_HOD.dat')
    plt.plot(data_in[:,0], data_in[:,2]/data_in[:,1], label='HOD', c='k')
    plt.xscale('log')
    plt.legend(framealpha=0)
    plt.xlabel(label_mass)
    plt.ylabel(r'$\sigma_{\lambda}/\lambda$')
    plt.savefig(loc+'sigma_lambda_ov_lambda.png')
    plt.show()

def fit_mass_scatter():
    for i in range(len(fname_kllr)):
        data_in = np.loadtxt(loc+fname_kllr[i])
        mass_fit = data_in[:,0]
        sigma_lambda_ov_lambda_fit = data_in[:,2]/data_in[:,1]
        
        sel = (mass_fit > 0)
        mass_fit = mass_fit[sel]
        sigma_lambda_ov_lambda_fit = sigma_lambda_ov_lambda_fit[sel]
        
        popt = curve_fit(murata2018, mass_fit, sigma_lambda_ov_lambda_fit)[0]
        plt.plot(mass_fit, sigma_lambda_ov_lambda_fit, label=label_kllr[i])
        plt.plot(mass_fit, murata2018(mass_fit, popt[0], popt[1]))
        plt.legend()
        plt.title(r'$\rm \sigma_0 = '+f'{popt[0]:.4f}, q='+f'{popt[1]:.4f}$')
        plt.xscale('log')
        plt.show()

if __name__=='__main__':
    plot_projection_component()
    plot_f_proj()
    plot_sigma_cl()
    plot_redshift_scan()
    plot_richness_mass()
    plot_n_greater_lambda()
    plot_sigma_lambda_ov_lambda()
    fit_mass_scatter()