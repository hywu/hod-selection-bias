#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

lam_min_list = np.array([20, 30, 45, 60])
lam_max_list = np.array([30, 45, 60, 1000])
nbins = len(lam_min_list)

def plot_lensing_y1(thresholded, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, nbins, figsize=(20, 5))

    if thresholded == True:
        bin_or_threshold = 'threshold'
        marker = 'x'
    else:
        bin_or_threshold = 'bin'
        marker = 'o'

    for ibin in range(nbins):
        out_loc = '/bsuhome/hwu/work/hod/output/plots_for_paper/y1/'
        rp, DS, dDS = np.loadtxt(f'{out_loc}/y1_DS_{bin_or_threshold}_z_0.2_0.35_lam_{ibin}.dat', unpack=True)
        sel = (DS > 0)&(rp >= 0.25)
        ax = axes[ibin]
        ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2, markersize=12, marker=marker, mec='k')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\rm r_p [pMpc]$')
        ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')

    '''
    if thresholded == True:
        for ibin in range(nbins):
            out_loc = '/bsuhome/hwu/work/hod/output/plots_for_paper/y1/'
            rp, DS, dDS = np.loadtxt(f'{out_loc}/DS_threshold_{ibin}.dat', unpack=True)
            sel = (DS > 0)&(rp >= 0.25)
            ax = axes[ibin]
            ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2, markersize=12, marker='x', mec='k')

            ax.set_xscale('log')
            ax.set_xlabel(r'$\rm r_p [pMpc]$')
            ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')
    else:
        for ibin in range(nbins):
            loc = '/bsuhome/hwu/scratch/hod-selection-bias/des_y1/DESY1_CL_WL/'
            z_bin = 0
            lam_bin = ibin + 3
            fname = loc + f'full-unblind-v2-mcal-zmix_y1subtr_l{lam_bin}_z{z_bin}_profile.dat'
            # R [Mpc]       DeltaSigma_t [M_sun / pc^2]     DeltaSigma_t_err [M_sun / pc^2] DeltaSigma_x [M_sun / pc^2]     DeltaSigma_x_err [M_sun / pc^2]
            data = np.loadtxt(fname)
            rp = data[:,0]
            DS = data[:,1]
            dDS = data[:,2]
            sel = (DS > 0)&(rp >= 0.25)
            ax = axes[ibin]
            ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2 , marker='o', mec='k')#, label='Y1 bin %i'%(ibin))
            
            ## Boost factor 
            # fname2 = loc + f'full-unblind-v2-mcal-zmix_y1clust_l{lam_bin}_z{z_bin}_zpdf_boost.dat'
            # # R     1+B(z)  B(z)_err
            # data2 = np.loadtxt(fname2)
            # rp_boost = data2[:,0]
            # boost = data2[:,1]
            # boost_interp = interp1d(rp_boost, boost)
            # sel = (rp >= min(rp_boost))&(rp <= max(rp_boost))
            #ax.errorbar(rp[sel], rp[sel]*DS[sel]*boost_interp(rp[sel]), rp[sel]*dDS[sel]*boost_interp(rp[sel]), capsize=8, c='gray', ls='', elinewidth=2 , marker='o', mec='gray')#, label='boosted')
            #ax.legend()
            ax.set_xscale('log')
            ax.set_xlabel(r'$\rm r_p\ [pMpc]$')
            ax.set_ylabel(r'$\rm r_p \Delta\Sigma\ [pMpc\ M_\odot/ppc^2]$')
            '''