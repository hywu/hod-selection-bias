#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import fitsio
import h5py
import os
import sys
import yaml
sys.path.append('../utils')
from measure_lensing import MeasureLensing
from sample_matching_mass import sample_matching_mass

lam_min_list = np.array([20, 30, 45, 60])
lam_max_list = np.array([30, 45, 60, 1000])
nbins = len(lam_min_list)

class PlotLensing(object):
    def __init__(self, yml_fname, abundance_matching, thresholded):

        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.Rmin = para.get('Rmin', 0.01)
        self.Rmax = para.get('Rmax', 100)
        self.pimax = para.get('pimax', 100)
        self.depth = para['depth']

        redshift = para['redshift']
        self.scale_factor = 1/(1+redshift)

        output_loc = para['output_loc']
        model_name = para['model_name']
        self.out_path = f'{output_loc}/model_{model_name}'

        use_pmem = para.get('use_pmem', False)
        self.los = para.get('los', 'z')
        if use_pmem == True:
            self.rich_name = f'pmem'
        else:
            self.rich_name = f'd{self.depth:.0f}'

        if self.los != 'z':
            self.rich_name = f'{self.rich_name}_{self.los}'

        if para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            self.readcat = ReadMiniUchuu(para['nbody_loc'])

        if para['nbody'] == 'abacus_summit':
            from read_abacus_summit import ReadAbacusSummit
            self.readcat = ReadAbacusSummit(para['nbody_loc'])

        if para['nbody'] == 'tng_dmo':
            from read_tng_dmo import ReadTNGDMO
            halofinder = para.get('halofinder', 'rockstar')
            self.readcat = ReadTNGDMO(para['nbody_loc'], halofinder)
            print('halofinder', halofinder)

        self.mpart = self.readcat.mpart
        self.boxsize = self.readcat.boxsize
        self.hubble = self.readcat.hubble
        self.vol = self.boxsize**3

        if os.path.isdir(f'{self.out_path}/obs_{self.rich_name}/')==False: 
            os.makedirs(f'{self.out_path}/obs_{self.rich_name}/')

        if abundance_matching == True:
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

        self.fname1 = f'{self.out_path}/obs_{self.rich_name}/Sigma_{self.outname}'
        self.fname2 = f'{self.out_path}/obs_{self.rich_name}/DS_{self.outname}'
        self.fname3 = f'{self.out_path}/obs_{self.rich_name}/mass_{self.outname}'
        self.fname4 = f'{self.out_path}/obs_{self.rich_name}/lam_{self.outname}'

    def calc_lensing(self):
        xp_in, yp_in, zp_in = self.readcat.read_particles()
        if self.los == 'z':
            self.xp = xp_in
            self.yp = yp_in
            self.zp = zp_in
        if self.los == 'x':
            self.xp = yp_in
            self.yp = zp_in
            self.zp = xp_in
        if self.los == 'y':
            self.xp = zp_in
            self.yp = xp_in
            self.zp = yp_in
        print('finished reading particles')

        # get clusters
        fname = f'{self.out_path}/richness_{self.rich_name}.fit'
        data, header = fitsio.read(fname, header=True)
        xh_all = data['px'] # already rotated when calculating richness
        yh_all = data['py']
        zh_all = data['pz']
        lnM_all = np.log(data['M200m'])
        lam_all = data['lambda']

        # shuffle the sample to remove mass sorting
        shuff = np.random.permutation(len(xh_all))
        xh_all = xh_all[shuff]
        yh_all = yh_all[shuff]
        zh_all = zh_all[shuff]
        lnM_all = lnM_all[shuff]
        lam_all = lam_all[shuff]

        if self.abundance_matching == True:
            sort = np.argsort(-lam_all)
            xh_all = xh_all[sort]
            yh_all = yh_all[sort]
            zh_all = zh_all[sort]
            lnM_all = lnM_all[sort]
            lam_all = lam_all[sort]

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
                lam_sel = lam_all[sel]

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
                lam_sel = lam_all[counts_max:counts_min]

            out_loc = f'{self.out_path}/obs_{self.rich_name}/'

            ml = MeasureLensing(out_loc, self.Rmin, self.Rmax, self.pimax)
            ml.write_bin_file()
            xh_mat, yh_mat, zh_mat, lnM_matched = sample_matching_mass(lnM_sel, lnM_all, xh_all, yh_all, zh_all)
            rp, Sigma_sel, DS_sel = ml.measure_lensing(xh_sel, yh_sel, zh_sel, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
            rp, Sigma_mat, DS_mat = ml.measure_lensing(xh_mat, yh_mat, zh_mat, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
            sel = (rp > 0.1)
            x = rp[sel]
            y = Sigma_sel[sel]
            z = Sigma_mat[sel]
            data1 = np.array([x,y,z]).transpose()
            np.savetxt(f'{self.fname1}_{ibin}.dat', data1, fmt='%-12g', header='rp, Sigma_sel, Sigma_matched')

            x = rp[sel]
            y = DS_sel[sel]
            z = DS_mat[sel]
            data2 = np.array([x,y,z]).transpose()
            np.savetxt(f'{self.fname2}_{ibin}.dat', data2, fmt='%-12g', header='rp, DS_sel, DS_matched')

            # these two files are for sanity checks
            data3 = np.array([lnM_sel]).transpose()
            np.savetxt(f'{self.fname3}_{ibin}.dat', data3, fmt='%-12g', header='lnM200m')

            data4 = np.array([lam_sel]).transpose()
            np.savetxt(f'{self.fname4}_{ibin}.dat', data4, fmt='%-12g', header='lambda')


    def plot_lensing(self, axes=None, label=None, plot_bias=False, color=None, lw=None):
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

            if plot_bias == False:
                try:
                    rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                    rp = rp / self.hubble * self.scale_factor
                    DS_sel = DS_sel * 1e-12 * self.hubble / self.scale_factor**2
                    ax = axes[ibin]
                    ax.plot(rp, rp*DS_sel, label=label, color=color, lw=lw)
                    ax.set_xscale('log')
                    ax.set_title(title)
                    if ibin == 0: # nbins - 1: 
                        ax.legend()
                    ax.set_xlabel(r'$\rm r_p \ [pMpc]$')
                    ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')
                except:
                    print(f'need to run {self.fname2}_{ibin}.dat')

            if plot_bias == True:
                rp, Sigma_sel, Sigma_matched = np.loadtxt(f'{self.fname1}_{ibin}.dat', unpack=True)
                rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                Sigma_bias = Sigma_sel / Sigma_matched
                DS_bias = DS_sel / DS_matched

                ax = axes[0,ibin]
                ax.semilogx(rp, Sigma_bias, label=label, color=color, lw=lw)
                ax.set_title(title)
                ax.legend()
                ax.set_xlabel(r'$\rm r_p$')
                ax.set_ylabel(r'$\Sigma$ bias')
                ax.axhline(1, c='gray', ls='--')

                ax = axes[1,ibin]
                ax.semilogx(rp, DS_bias, label=label)
                ax.set_title(title)
                ax.legend()
                ax.set_xlabel(r'$\rm r_p$')
                ax.set_ylabel(r'$\Delta\Sigma$ bias')
                ax.axhline(1, c='gray', ls='--')


if __name__ == "__main__":
    yml_fname = sys.argv[1]
    plmu = PlotLensing(yml_fname, abundance_matching=True, thresholded=False)
    plmu.calc_lensing()
    #plmu.plot_lensing()
    #plt.show()