#!/usr/bin/env python
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import fitsio
import os, sys
import yaml
from scipy.interpolate import interp1d
from astropy.cosmology import w0waCDM
import timeit
start = timeit.default_timer()
start_master = start * 1

#sys.path.append('../utils')
from hod.utils.read_sim import read_sim
from hod.utils.measure_lensing import MeasureLensing
from hod.utils.sample_matching_mass import sample_matching_mass
from hod.utils.print_memory import print_memory
from hod.utils.get_para_abacus_summit import get_cosmo_para, get_hod_para


class PlotLensing(object):
    def __init__(self, yml_fname, binning, thresholded=False):
        # binning: 'abun', 'lam', 'AB'

        self.binning = binning
        self.thresholded = thresholded

        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.Rmin = para.get('Rmin', 0.01)
        self.Rmax = para.get('Rmax', 100)
        self.pimax = para.get('pimax', 100)
        self.nrp_per_decade = para.get('nrp_per_decade', 5)

        seed = para.get('seed', 42)
        self.rng = default_rng(seed)

        self.redshift = para['redshift']
        self.scale_factor = 1./ (1. + self.redshift)

        self.survey = para.get('survey', 'desy1')

        if self.survey == 'desy1':
            self.lam_min_list = np.array([20, 30, 45, 60])
            self.lam_max_list = np.array([30, 45, 60, 1000])
            self.nbins = len(self.lam_min_list)

        if self.survey == 'desy1thre':
            self.lam_min_list = np.array([20])
            self.lam_max_list = np.array([1000])
            self.nbins = len(self.lam_min_list)

        if self.survey == 'sdss':
            self.lam_min_list = np.array([5])
            self.lam_max_list = np.array([140])
            self.nbins = len(self.lam_min_list)

        if self.survey == 'sdssbins':
            self.lam_min_list = np.array([20, 30, 45, 60])
            self.lam_max_list = np.array([30, 45, 60, 1000])
            self.nbins = len(self.lam_min_list)

        #### For AbacusSummit ####
        if para['nbody'] == 'abacus_summit':
            cosmo_id = para.get('cosmo_id', None)
            hod_id = para.get('hod_id', None)
            phase = para.get('phase', None)
            if self.redshift == 0.3: z_str = '0p300'
            if self.redshift == 0.4: z_str = '0p400'
            if self.redshift == 0.5: z_str = '0p500'
            output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'

            cosmo_abacus = get_cosmo_para(cosmo_id)
            self.Om0 = cosmo_abacus['OmegaM']
            self.w0 = cosmo_abacus['w0']
            self.wa = cosmo_abacus['wa']
            
            if binning == 'AB':
                hod_para = get_hod_para(hod_id)
                self.A = hod_para['A']
                self.B = hod_para['B']
        else:
            output_loc = para['output_loc']
            self.Om0 = para['OmegaM']
            self.w0 = para.get('w0', -1)
            self.wa = para.get('wa', 0)

        model_name = para['model_name']
        self.out_path = f'{output_loc}/model_{model_name}'
        self.rich_name = para['rich_name']

        self.los = para.get('los', 'z')
        if self.los != 'z':
            self.rich_name = f'{self.rich_name}_{self.los}'

        self.readcat = read_sim(para)
        self.mpart = self.readcat.mpart
        self.boxsize = self.readcat.boxsize
        self.hubble = self.readcat.hubble
        self.vol = self.boxsize**3

        '''
        if self.binning == 'abundance_matching':
            self.outname = 'abun'
        if self.binning == 'Ncyl':
            self.outname = 'lam'
        if self.binning == 'AB_scaling':
            self.outname = 'AB'
        '''
        self.outname = self.binning

        if thresholded == True:
            self.outname += '_thre'
        else:
            self.outname += '_bin'

        self.obs_path = f'{self.out_path}/obs_{self.rich_name}_{self.survey}/'
        if os.path.isdir(self.obs_path)==False: 
            os.makedirs(self.obs_path)

        self.fname1 = f'{self.obs_path}/Sigma_{self.outname}'
        self.fname2 = f'{self.obs_path}/DS_{self.outname}'
        self.fname3 = f'{self.obs_path}/mass_{self.outname}'
        self.fname4 = f'{self.obs_path}/lam_{self.outname}'
        self.fname5 = f'{self.obs_path}/DS_phys_noh_{self.outname}' ## NEW!
        print('lens saved at', self.fname5)


    def set_up_abundance_matching(self):
        # calculate expected counts in Abacus, assuming current cosmology
        if self.survey == 'desy1' or self.survey == 'desy1thre':
            survey_area_sq_deg = 1437
            if self.redshift == 0.3:
                 zmin = 0.2 
                 zmax = 0.35
            #### Y1 volume assuming this cosmology ####
            fsky = survey_area_sq_deg/41253.
            Ode0 = 1 - self.Om0
            cosmo = w0waCDM(H0=100, Om0=self.Om0, Ode0=Ode0, w0=self.w0, wa=self.wa)
            desy1_vol = fsky * 4. * np.pi/3. * (cosmo.comoving_distance(zmax).value**3 - cosmo.comoving_distance(zmin).value**3) # (Mpc/h)**3
            # TODO: use redmapper area to get the accurate volume
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            out_loc = os.path.join(BASE_DIR, '../y1/data/')
            #if self.redshift == 0.3: 
            cum_counts = np.loadtxt(out_loc+f'cluster_cumulative_counts_no_miscen_z_{zmin}_{zmax}.dat')
            cum_den = cum_counts/desy1_vol

        counts_list = np.array(np.around(cum_den * self.vol)+1e-4, dtype=int)
        counts_list = np.append(counts_list, 0)
        self.counts_min_list = counts_list[0:-1]
        self.counts_max_list = counts_list[1:]
        
    def read_particles(self):
        print_memory('before reading particles')
        xp_in, yp_in, zp_in = self.readcat.read_particle_positions()
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
        print_memory('finished reading particles')



    def calc_lensing(self):

        # get clusters
        fname = f'{self.out_path}/richness_{self.rich_name}.fit'
        fname2 = f'{self.out_path}/richness_{self.rich_name}_best_match.fit'
        
        if os.path.exists(fname2):
            print('use best_match mass')
            data, header = fitsio.read(fname2, header=True)
            mass_all = data['mass_best_match']
        else:
            data, header = fitsio.read(fname, header=True)
            mass_all = data['mass_host']

        xh_all = data['px']
        yh_all = data['py']
        zh_all = data['pz']
        lnM_all = np.log(mass_all)
        lam_all = data['lambda']

        # shuffle the sample to remove mass sorting
        shuff = self.rng.permutation(len(xh_all))
        xh_all = xh_all[shuff]
        yh_all = yh_all[shuff]
        zh_all = zh_all[shuff]
        lnM_all = lnM_all[shuff]
        lam_all = lam_all[shuff]

        if self.binning == 'AB':
            lnlam_new = self.A * np.log(lam_all) + self.B
            lam_all = np.exp(lnlam_new)

        if self.binning == 'abun': #only sort mass if abun match
            self.set_up_abundance_matching()
            sort = np.argsort(-lam_all)
            xh_all = xh_all[sort]
            yh_all = yh_all[sort]
            zh_all = zh_all[sort]
            lnM_all = lnM_all[sort]
            lam_all = lam_all[sort]

        for ibin in range(self.nbins):
            if os.path.exists(f'{self.fname5}_{ibin}.dat'):
                print('done', f'{self.fname5}_{ibin}.dat')
            else:
                print_memory(f'doing bin{ibin}')
                if self.binning == 'lam' or self.binning == 'AB':
                    lam_min = self.lam_min_list[ibin]
                    lam_max = self.lam_max_list[ibin]
                    if self.thresholded == True:
                        sel = (lam_all >= lam_min)
                    else:
                        sel = (lam_all >= lam_min)&(lam_all < lam_max)

                    xh_sel = xh_all[sel]
                    yh_sel = yh_all[sel]
                    zh_sel = zh_all[sel]
                    lnM_sel = lnM_all[sel]
                    lam_sel = lam_all[sel]

                if self.binning == 'abun':
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


                #out_loc = f'{self.out_path}/obs_{self.rich_name}/'

                ml = MeasureLensing(self.obs_path, self.Rmin, self.Rmax, self.pimax, self.nrp_per_decade)
                ml.write_bin_file()
                xh_mat, yh_mat, zh_mat, lnM_matched = sample_matching_mass(lnM_sel, lnM_all, xh_all, yh_all, zh_all)
                rp, Sigma_sel, DS_sel = ml.measure_lensing(xh_sel, yh_sel, zh_sel, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
                rp, Sigma_mat, DS_mat = ml.measure_lensing(xh_mat, yh_mat, zh_mat, self.xp, self.yp, self.zp, self.boxsize, self.mpart)
                sel = (rp > 0.05)
                print('saving', self.fname1)
                x = rp[sel]
                y = Sigma_sel[sel]
                z = Sigma_mat[sel]
                data1 = np.array([x,y,z]).transpose()
                np.savetxt(f'{self.fname1}_{ibin}.dat', data1, fmt='%-12g', header='rp [cMpc/h], Sigma_sel [h Msun/pc^2], Sigma_matched')

                x = rp[sel]
                y = DS_sel[sel]
                z = DS_mat[sel]
                data2 = np.array([x,y,z]).transpose()
                np.savetxt(f'{self.fname2}_{ibin}.dat', data2, fmt='%-12g', header='rp [cMpc/h], DS_sel [h Msun/pc^2], DS_matched')

                # extra: save physical no-h units, used for emulator
                x = rp[sel] / self.hubble * self.scale_factor
                y = DS_sel[sel] * self.hubble / self.scale_factor**2
                data5 = np.array([x,y]).transpose()
                np.savetxt(f'{self.fname5}_{ibin}.dat', data5, fmt='%-12g', header='rp [pMpc], DS [Msun/pc^2]')

                # these two files are for sanity checks
                data3 = np.array([lnM_sel]).transpose()
                np.savetxt(f'{self.fname3}_{ibin}.dat', data3, fmt='%-12g', header='lnMass [Msun/h]')

                data4 = np.array([lam_sel]).transpose()
                np.savetxt(f'{self.fname4}_{ibin}.dat', data4, fmt='%-12g', header='lambda')

    def plot_mass_pdf(self, axes=None, label=None, plot_bias=False, color=None, lw=None):
        if axes is None and plot_bias==False:
            fig, axes = plt.subplots(1, self.nbins, figsize=(20, 5))

        if axes is None and plot_bias==True:
            fig, axes = plt.subplots(1, self.nbins, figsize=(20, 5))

        for ibin in range(self.nbins):
            lnM = np.loadtxt(f'{self.fname3}_{ibin}.dat')
            ax = axes[ibin]
            ax.hist(lnM/np.log(10.), density=True, fc='none', 
                histtype='step', label=len(lnM), lw=lw)
            if label != None:
                ax.legend()
            if ibin == 0: # self.nbins - 1: 
                plt.title(label)
            ax.set_xlabel(r'$\rm  \log_{10} M$')
            ax.set_ylabel(r'PDF')

    def plot_lensing(self, axes=None, label=None, plot_bias=False, color=None, lw=None):
        if axes is None and plot_bias==False:
            fig, axes = plt.subplots(1, self.nbins, figsize=(20, 5))

        if axes is None and plot_bias==True:
            fig, axes = plt.subplots(1, self.nbins, figsize=(20, 5))

        for ibin in range(self.nbins):
            if self.binning == 'lam' or self.binning == 'AB':
                lam_min = self.lam_min_list[ibin]
                if self.thresholded == True:
                    title = r'$\lambda>%g$'%lam_min
                else:
                    lam_max = self.lam_max_list[ibin]
                    title = r'$%g < \lambda < %g$'%(lam_min, lam_max)

            if self.binning == 'abun':
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
                    DS_sel = DS_sel * self.hubble / self.scale_factor**2
                    ax = axes[ibin]
                    ax.plot(rp, rp*DS_sel, label=label, color=color, lw=lw)
                    ax.set_xscale('log')
                    ax.set_title(title)
                    if ibin == 0 and label!=None: # self.nbins - 1: 
                        ax.legend()
                    ax.set_xlabel(r'$\rm r_p \ [pMpc]$')
                    ax.set_ylabel(r'$\rm r_p \Delta\Sigma [pMpc M_\odot/ppc^2]$')
                except:
                    print(f'need to run {self.fname2}_{ibin}.dat')

            if plot_bias == True:
                #rp, Sigma_sel, Sigma_matched = np.loadtxt(f'{self.fname1}_{ibin}.dat', unpack=True)
                rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                #Sigma_bias = Sigma_sel / Sigma_matched
                DS_bias = DS_sel / DS_matched

                # ax = axes[0,ibin]
                # ax.semilogx(rp, Sigma_bias, label=label, color=color, lw=lw)
                # ax.set_title(title)
                # ax.legend()
                # ax.set_xlabel(r'$\rm r_p$')
                # ax.set_ylabel(r'$\Sigma$ bias')
                # ax.axhline(1, c='gray', ls='--')

                ax = axes[ibin]
                ax.semilogx(rp, DS_bias, label=label, color=color, lw=lw)
                ax.set_title(title)
                if label != None:
                    ax.legend()
                ax.set_xlabel(r'$\rm r_p$')
                ax.set_ylabel(r'$\Delta\Sigma$ bias')
                ax.axhline(1, c='gray', ls='--')


    def get_lensing_data_vector(self, get_bias=False): # use Y1 radius!
        rp_all = []
        DS_all = []
        for ibin in range(self.nbins):

            fname = f'../y1/data/y1_DS_bin_z_0.2_0.35_lam_{ibin}.dat'
            rp_y1 = np.loadtxt(fname)[:,0]

            if self.binning == 'lam':
                lam_min = self.lam_min_list[ibin]
                if self.thresholded == True:
                    lam_max = 1e6
                else:
                    lam_max = self.lam_max_list[ibin]

            if self.binning == 'abun':
                counts_min = self.counts_min_list[ibin]
                if self.thresholded == True:
                    space_density = counts_min / self.vol
                else:
                    counts_max = self.counts_max_list[ibin]
                    space_density_min = counts_min / self.vol
                    space_density_max = counts_max / self.vol

            if get_bias == False:
                if True:
                    rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
                    rp = rp / self.hubble * self.scale_factor
                    DS_sel = DS_sel * 1e-12 * self.hubble / self.scale_factor**2
                    #print(DS_sel)
                    DS_interp = interp1d(rp, DS_sel)
                    rp_all.append(rp_y1)
                    DS_all.append(DS_interp(rp_y1))



                # except:
                #     print(f'need to run {self.fname2}_{ibin}.dat')

            # if get_bias == True: # need to code this?
            #     try:
            #         rp, Sigma_sel, Sigma_matched = np.loadtxt(f'{self.fname1}_{ibin}.dat', unpack=True)
            #         rp, DS_sel, DS_matched = np.loadtxt(f'{self.fname2}_{ibin}.dat', unpack=True)
            #         Sigma_bias = Sigma_sel / Sigma_matched
            #         DS_bias = DS_sel / DS_matched
            #         rp_all.append(rp)
            #         DS_all.apppend(DS_bias)
            #     except:
            #         print(f'need to run {self.fname1}_{ibin}.dat')
                    
        return np.array(rp_all), np.array(DS_all)

if __name__ == "__main__":

    #./plot_lensing.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml

    yml_fname = sys.argv[1]
    binning = sys.argv[2]
    thresholded = sys.argv[3]
    plmu = PlotLensing(yml_fname, binning, thresholded)
    plmu.read_particles()
    plmu.calc_lensing()

    #plmu.plot_lensing()
    #plt.show()
    # rp, DS = plmu.get_lensing_data_vector()
    # print(np.shape(rp))

    stop = timeit.default_timer()
    dtime = (stop - start_master)/60.
    print(f'total time {dtime:.2g} mins')