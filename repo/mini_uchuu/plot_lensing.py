#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import fitsio, h5py
import os, sys

# my functions 
sys.path.append('../utils')
from measure_lensing import measure_lensing
from sample_matching_mass import sample_matching_mass
# sys.path.append('../uchuu/')
# from readGadgetSnapshot import readGadgetSnapshot

#### read in mini uchuu ####
# get header
# loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
# snap_header = readGadgetSnapshot(loc+f'snapdir_043/MiniUchuu_043.gad.0')

# hubble = snap_header.HubbleParam 
# redshift = snap_header.redshift
redshift = 0.3
scale_factor = 1/(1+redshift)


# data = fitsio.read(loc+'particles_043_0.1percent.fit')
# self.xp = data['x']
# yp = data['y']
# zp = data['z']
# mpart = snap_header.mass[1] * 1e10 * 1000
# boxsize = snap_header.BoxSize

# get the counts information
lam_min_list = np.array([20, 30, 45, 60])
lam_max_list = np.array([30, 45, 60, 1000])
nbins = len(lam_min_list)




class PlotLensing(object):
    def __init__(self, which_sim, model_id, depth, abundance_matching, thresholded):
        self.depth = depth

        if which_sim == 'mini_uchuu':
            output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
            from read_mini_uchuu import ReadMiniUchuu
            rmu = ReadMiniUchuu()
            self.xp, self.yp, self.zp = rmu.read_particles()
            self.mpart = rmu.mpart
            self.boxsize = rmu.boxsize
            self.hubble = rmu.hubble
            self.vol = self.boxsize**3

        hod_name = f'model_{model_id}'
        self.out_path = f'{output_loc}/{hod_name}/'

        if os.path.isdir(f'{self.out_path}/obs_d{self.depth:.0f}/')==False: 
            os.makedirs(f'{self.out_path}/obs_d{self.depth:.0f}/')

        if abundance_matching == True:
            
            #loc = '/bsuhome/hwu/work/hod/output/plots_for_paper/y1/'
            cum_den = np.loadtxt('../y1/data/cluster_cumulative_density_z_0.2_0.35.dat')
            counts_list = np.array(np.around(cum_den * self.vol)+1e-4, dtype=int)
            counts_list = np.append(counts_list, 0)
            self.counts_min_list = counts_list[0:-1]
            self.counts_max_list = counts_list[1:]

        self.abundance_matching = abundance_matching
        self.thresholded = thresholded

        if self.abundance_matching == True:
            self.outname = 'abun'
        else:
            self.outname = 'lam'

        if thresholded == True:
            self.outname += '_thre'
        else:
            self.outname += '_bin'

        self.fname1 = f'{self.out_path}/obs_d{self.depth:.0f}/Sigma_{self.outname}'
        self.fname2 = f'{self.out_path}/obs_d{self.depth:.0f}/DS_{self.outname}'
        self.fname3 = f'{self.out_path}/obs_d{self.depth:.0f}/mass_{self.outname}'

    def calc_lensing(self):
        # get clusters
        fname = f'{self.out_path}/richness_d{self.depth:.0f}.fit'
        data, header = fitsio.read(fname, header=True)
        xh_all = data['px']
        yh_all = data['py']
        zh_all = data['pz']
        lnM_all = np.log(data['M200m'])
        lam_all = data['lambda']

        sort = np.argsort(lnM_all) # from small to large.  Shouldn't make a difference.  # change it to shuffle
        xh_all = xh_all[sort]
        yh_all = yh_all[sort]
        zh_all = zh_all[sort]
        lnM_all = lnM_all[sort]
        lam_all = lam_all[sort]

        if self.abundance_matching == True:
            sort = np.argsort(-lam_all)
            xh_all = xh_all[sort]
            yh_all = yh_all[sort]
            zh_all = zh_all[sort]
            lnM_all = lnM_all[sort]

        for ibin in range(nbins):
            if self.abundance_matching == False:
                lam_min = lam_min_list[ibin]
                lam_max = lam_max_list[ibin]
                if self.thresholded == True:
                    sel = (lam_all >= lam_min)
                else:
                    sel = (lam_all >= lam_min)&(lam_all < lam_max)

                xh_sel = xh_all[sel]
                yh_sel = yh_all[sel]
                zh_sel = zh_all[sel]
                lnM_sel = lnM_all[sel]

            if self.abundance_matching == True:
                if self.thresholded == True:
                    counts_min = self.counts_min_list[ibin]
                    counts_max = 0
                else:
                    counts_min = self.counts_min_list[ibin]
                    counts_max = self.counts_max_list[ibin]

                xh_sel = xh_all[counts_max:counts_min]
                yh_sel = yh_all[counts_max:counts_min]
                zh_sel = zh_all[counts_max:counts_min]
                lnM_sel = lnM_all[counts_max:counts_min]

            #print('n clusters = ', len(xh_sel))
            xh_mat, yh_mat, zh_mat, lnM_matched = sample_matching_mass(lnM_sel, lnM_all, xh_all, yh_all, zh_all)
            rp, Sigma_sel, DS_sel = measure_lensing(xh_sel, yh_sel, zh_sel, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
            rp, Sigma_mat, DS_mat = measure_lensing(xh_mat, yh_mat, zh_mat, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
            sel = (rp > 0.1)
            x = rp[sel]
            y = Sigma_sel[sel]
            z = Sigma_mat[sel]
            data = np.array([x,y,z]).transpose()
            np.savetxt(f'{self.fname1}_{ibin}.dat', data, fmt='%-12g', header='rp, Sigma_sel, Sigma_matched')

            x = rp[sel]
            y = DS_sel[sel]
            z = DS_mat[sel]
            data = np.array([x,y,z]).transpose()
            np.savetxt(f'{self.fname2}_{ibin}.dat', data, fmt='%-12g', header='rp, DS_sel, DS_matched')

            data3 = np.array([lnM_sel]).transpose()
            np.savetxt(f'{self.fname3}_{ibin}.dat', data3, fmt='%-12g', header='lnM200m')


    def plot_lensing(self, axes=None, label=None, plot_bias=False):
        if axes is None and plot_bias==False:
            fig, axes = plt.subplots(1, nbins, figsize=(20, 5))

        if axes is None and plot_bias==True:
            fig, axes = plt.subplots(2, nbins, figsize=(20, 10))

        for ibin in range(nbins):
            if self.abundance_matching == False:
                lam_min = lam_min_list[ibin]
                if self.thresholded == True:
                    title = r'$\lambda>%g$'%lam_min
                else:
                    lam_max = lam_max_list[ibin]
                    title = r'$%g < \lambda < %g$'%(lam_min, lam_max)

            if self.abundance_matching == True:
                counts_min = self.counts_min_list[ibin]
                if self.thresholded == True:
                    space_density = counts_min / self.vol
                    title = f'space density = {space_density:.0e}'
                else:
                    counts_max = self.counts_max_list[ibin]
                    space_density_min = counts_min / self.vol
                    space_density_max = counts_max / self.vol
                    title = f'space density: {space_density_min:.2e} to {space_density_max:.2e}'

            # if os.path.exists(self.fname1) == False or os.path.exists(self.fname2) == False:
            #     print('calc lensing')
            #     self.calc_lensing()
                
            if plot_bias == False:
                rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                rp = rp / self.hubble * scale_factor
                DS_sel = DS_sel * 1e-12 * self.hubble / scale_factor**2
                ax = axes[ibin]
                ax.plot(rp, rp*DS_sel, label=label)
                ax.set_xscale('log')
                ax.set_title(title)
                if ibin == nbins - 1: ax.legend()
                ax.set_xlabel(r'$\rm r_p \ [pMpc]$')
                ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')

            if plot_bias == True:
                rp, Sigma_sel, Sigma_matched = np.loadtxt(f'{self.fname1}_{ibin}.dat', unpack=True)
                rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                Sigma_bias = Sigma_sel / Sigma_matched
                DS_bias = DS_sel / DS_matched

                ax = axes[0,ibin]
                ax.semilogx(rp, Sigma_bias, label=label)
                ax.set_title(title)
                ax.legend()
                ax.set_xlabel(r'$\rm r_p$')
                ax.set_ylabel(r'$\Sigma$ bias')
                #ax.set_ylim(0.9, 1.2)
                ax.axhline(1, c='gray', ls='--')

                ax = axes[1,ibin]
                ax.semilogx(rp, DS_bias, label=label)
                ax.set_title(title)
                ax.legend()
                ax.set_xlabel(r'$\rm r_p$')
                ax.set_ylabel(r'$\Delta\Sigma$ bias')
                #ax.set_ylim(0.9, 1.5)
                ax.axhline(1, c='gray', ls='--')


if __name__ == "__main__":
    plmu = PlotLensing('mini_uchuu', 0, 30, abundance_matching=True, thresholded=False)
    plmu.calc_lensing()
    plmu.plot_lensing()
    plt.show()