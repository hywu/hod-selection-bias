#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

zmin_list = [0.2, 0.35, 0.5]
zmax_list = [0.35, 0.5, 0.65]
lam_min_list = np.array([20, 30, 45, 60])
lam_max_list = np.array([30, 45, 60, 1000])
nbins = len(lam_min_list)

def plot_lensing_y1(iz, thresholded, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, nbins, figsize=(20, 5))

    if thresholded == True:
        bin_or_threshold = 'threshold'
        marker = 'x'
    else:
        bin_or_threshold = 'bin'
        marker = 'o'

    for ilam in range(nbins):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        out_loc = os.path.join(BASE_DIR, 'data')
        zmin = zmin_list[iz]
        zmax = zmax_list[iz]
        lam_min = lam_min_list[ilam]
        lam_max = lam_max_list[ilam]
        rp, DS, dDS = np.loadtxt(f'{out_loc}/y1_DS_{bin_or_threshold}_z_{zmin}_{zmax}_lam_{lam_min}_{lam_max}.dat', unpack=True)
        sel = (DS > 0)&(rp >= 0.25)
        ax = axes[ilam]
        ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2, markersize=12, marker=marker, mec='k')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\rm r_p [pMpc]$')
        ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')

    r'''
    if thresholded == True:
        for ilam in range(nbins):
            out_loc = '/bsuhome/hwu/work/hod/output/plots_for_paper/y1/'
            rp, DS, dDS = np.loadtxt(f'{out_loc}/DS_threshold_{ilam}.dat', unpack=True)
            sel = (DS > 0)&(rp >= 0.25)
            ax = axes[ilam]
            ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2, markersize=12, marker='x', mec='k')

            ax.set_xscale('log')
            ax.set_xlabel(r'$\rm r_p [pMpc]$')
            ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')
    else:
        for ilam in range(nbins):
            loc = '/bsuhome/hwu/scratch/hod-selection-bias/des_y1/DESY1_CL_WL/'
            z_bin = 0
            lam_bin = ilam + 3
            fname = loc + f'full-unblind-v2-mcal-zmix_y1subtr_l{lam_bin}_z{z_bin}_profile.dat'
            # R [Mpc]       DeltaSigma_t [M_sun / pc^2]     DeltaSigma_t_err [M_sun / pc^2] DeltaSigma_x [M_sun / pc^2]     DeltaSigma_x_err [M_sun / pc^2]
            data = np.loadtxt(fname)
            rp = data[:,0]
            DS = data[:,1]
            dDS = data[:,2]
            sel = (DS > 0)&(rp >= 0.25)
            ax = axes[ilam]
            ax.errorbar(rp[sel], rp[sel]*DS[sel], rp[sel]*dDS[sel], capsize=8, c='k', ls='', elinewidth=2 , marker='o', mec='k')#, label='Y1 bin %i'%(ilam))
            
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