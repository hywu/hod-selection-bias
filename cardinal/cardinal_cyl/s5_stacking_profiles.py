#!/usr/bin/env python
import numpy as np
import fitsio
import os, sys
import h5py
from astropy.table import Table
from astropy.table import QTable
from astropy.table import join
from astropy.table import join_distance

sys.path.append('/users/hywu/work/selection/repo/stacking_profiles_weighted/')
import sample_binning as sb
import stacked_profile_weighted_by_mass_redshift as spw

sys.path.append('/users/hywu/work/selection/repo/chinchilla_profiles/')
import radial_bins_phys_mpc as rbp
import radial_bins_3d as r3d

chisq_cut = int(sys.argv[1])#100
output_loc = '/projects/hywu/cluster_sims/cluster_finding/data/cardinal_cyl/'
output_loc += f'model_chisq{chisq_cut}/'

plotting = False


## TODO
## match "correctfile" and my richness. Done
## implement Ncyl binning. Done
## implement abundance matching


### volume ####
from volume_cardinal import volume_gold


class StackingProfiles(object):
    def __init__(self, zmin, zmax, obs_name='DeltaSigma', method='weighted'):
        self.zmin = zmin
        self.zmax = zmax
        self.obs_name = obs_name
        self.method = method

        self.vol = volume_gold(self.zmin, self.zmax)
        self.binning = 'abundance_matching'

        self.lam_min_list = np.array([20, 30, 45, 60])
        self.lam_max_list = np.array([30, 45, 60, 1000])
        self.nbins = len(self.lam_min_list)

        self.obs_path = f'{output_loc}/obs/'


    def read_halos(self):
        ## read in halo file and profiles
        card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
        fname = card_loc+'correctfilev2.npy'
        data = Table(np.load(fname, allow_pickle=True))
        z_all = np.array(data['Redshift'])
        sel_z = (z_all > self.zmin-0.01)&(z_all < self.zmax + 0.01) # to account for small z diff between the 2 catalogs
        if self.obs_name == 'Sigma':
            profile_all = data['Sigma'][sel_z]
        if self.obs_name == 'DeltaSigma':
            profile_all = data['DeltaSigma'][sel_z]
        if self.obs_name == 'rho':
            profile_all = data['rho'][sel_z]
        hid_all = np.array(data['haloid'][sel_z])
        Mvir_all = np.array(data['Mvir'][sel_z])
        lnMvir_all = np.log(Mvir_all)
        z_all = np.array(data['Redshift'][sel_z])
        lnMvir_all = np.around(lnMvir_all, decimals=1)
        z_all = np.around(z_all, decimals=1)

        ## read in the cluster file & cut redshift
        alt_rich_fname = output_loc + 'Ncyl.fit'
        data = fitsio.read(alt_rich_fname)
        z = data['z_cos'] # close enough to the z in correctfile
        sel_z = (z > self.zmin)&(z < self.zmax)
        z = data['z_cos'][sel_z]
        hid_cyl = data['id'][sel_z]
        Ncyl = data['Ncyl'][sel_z]
        lnMvir_cyl = np.log(data['Mvir'][sel_z])
        lnMvir_cyl = np.around(lnMvir_cyl, decimals=1)
        z = np.around(z, decimals=1)

        print('duplicated id in correct file?',  len(hid_all) - len(np.unique(hid_all))) # has duplicates
        print('len(hid_cyl)', len(hid_cyl)) # no duplicates? really?

        data1 = QTable([hid_all, lnMvir_all, z_all, profile_all], names=('hid', 'lnM', 'z', 'profile'))
        data2 = QTable([hid_cyl, lnMvir_cyl, Ncyl, z], names=('hid', 'lnM', 'Ncyl', 'z'))
        self.data_all = join(data1, data2, keys=['hid', 'z'])#, 
        print('np.shape(self.data_all)', np.shape(self.data_all))
        
        # TODO: What to do with the duplicated IDs?
        #self.data_all2 = join(data1, data2, keys=['hid'])#, 
        #print('np.shape(self.data_all2)', np.shape(self.data_all2))

        #join_funcs={'hid': join_distance(0), 'lnM': join_distance(0.01)}) #, 'lnM'
        # print(self.data_all.keys())
        #print(self.data_all[0:5])
        #diff = np.abs(self.data_all['lnM_1'] - self.data_all['lnM_2'])
        #print('mass diff', max(diff), min(diff))
        #exit()
        # all halos
        self.lnM_all = self.data_all['lnM_2']
        self.lam_all = self.data_all['Ncyl']
        self.z_all = self.data_all['z']
        self.profile_all = self.data_all['profile']

        if self.binning == 'abundance_matching':
            #if self.survey == 'desy1':
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            out_loc = os.path.join(BASE_DIR, '../../hod/y1/data/')
            #if redshift == 0.3: 
            cum_den = np.loadtxt(out_loc+f'cluster_cumulative_density_z_{self.zmin}_{self.zmax}.dat')
            counts_list = np.array(np.around(cum_den * self.vol)+1e-4, dtype=int)
            counts_list = np.append(counts_list, 0)
            self.counts_min_list = counts_list[0:-1]
            self.counts_max_list = counts_list[1:]
            print('counts_list', counts_list)
            print(len(self.lnM_all))


    def stacking_profiles(self):#, lam_min, lam_max, dm = 0.1, dz = 0.05):

        if self.binning == 'abundance_matching': #only sort mass if abun match
            sort = np.argsort(-self.lam_all)
            lnM_all = self.lnM_all[sort]
            lam_all = self.lam_all[sort]
            profile_all = self.profile_all[sort]

        for ibin in range(self.nbins):
            if self.binning == 'Ncyl' or self.binning == 'AB_scaling':
                lam_min = self.lam_min_list[ibin]
                lam_max = self.lam_max_list[ibin]
                if self.thresholded == True:
                    sel = (lam_all >= lam_min)
                else:
                    sel = (lam_all >= lam_min)&(lam_all < lam_max)

                # xh_sel = xh_all[sel]
                # yh_sel = yh_all[sel]
                # zh_sel = zh_all[sel]
                lnM_sel = lnM_all[sel]
                lam_sel = lam_all[sel]
                profile_select = profile_all[sel]

            if self.binning == 'abundance_matching':
                # if self.thresholded == True:
                #     counts_min = self.counts_min_list[ibin]
                #     counts_max = 0
                # else:
                counts_min = self.counts_min_list[ibin]
                counts_max = self.counts_max_list[ibin]

                # xh_sel = xh_all[counts_max:counts_min]
                # yh_sel = yh_all[counts_max:counts_min]
                # zh_sel = zh_all[counts_max:counts_min]
                lnM_sel = lnM_all[counts_max:counts_min]
                lam_sel = lam_all[counts_max:counts_min]
                profile_select = profile_all[counts_max:counts_min]
                print('counts_min, counts_max', counts_min, counts_max)

            profile_select_mean = np.mean(profile_select, axis=0)
            # save physical no-h units, used for emulator
            rp = rbp.rp_phys_mpc
            sel = (rp > 0.05)
            x = rp #rp[sel] / self.hubble * self.scale_factor
            y = profile_select_mean #DS_sel[sel] * self.hubble / self.scale_factor**2
            data5 = np.array([x,y]).transpose()
            
            fname = f'{self.obs_path}/DS_phys_noh_abun_z_{self.zmin}_{self.zmax}'
            print('lens saved at', fname)
            np.savetxt(f'{fname}_bin_{ibin}.dat', data5, fmt='%-12g', header='rp [pMpc], DS [Msun/pc^2]')

        '''
        Ncyl = self.data_all['Ncyl']
        z = self.data_all['z']

        select_obs = (Ncyl >= lam_min)&(Ncyl < lam_max)&(z >= zmin)&(z < zmax)



        # halos in this bin
        data_select = self.data_all[select_obs]
        lnMvir_select = data_select['lnM']
        z_select = data_select['z']
        profile_select = data_select['profile']
        profile_select_mean = np.mean(profile_select, axis=0)

        if self.method == 'weighted':
            profile_unbiased = spw.stacked_profile_weighted_by_mass_redshift(lnMvir_select, z_select, profile_select, self.lnMvir_all, self.z_all, self.profile_all, dm=dm, dz=dz)

        return profile_select_mean, profile_unbiased
        '''
    '''
    def run_all_bins(self, use_thresholds=False):

        #bin_fname = data_loc + 'bins.dat'
        #iz_list, i_lam_list, n_cl_list, alt_rich_min, alt_rich_max = np.loadtxt(bin_fname, unpack=True)

        #self.ofname = data_loc + '%s.dat'%(self.obs_name)
        #self.outname = 'abun_bin'
        #fname = f'{data_loc}/DS_phys_noh_{self.outname}' ## NEW!
        #outfile = open(self.ofname,'w')
        #outfile.write('#iz, iam, %s ratio \n'%(self.obs_name))

        for iz in range(len(sb.zmin_list)):
            for ilam in range(len(sb.lam_min_list)):
                profile_select_mean, profile_stacked = self.stacking_profiles(zmin=sb.zmin_list[iz], zmax=sb.zmax_list[iz], lam_min=alt_rich_min[iz*4+ilam], lam_max=alt_rich_max[iz*4+ilam])
                
                
                if self.obs_name == 'rho':
                    r = r3d.r
                else:
                    r = rbp.rp_phys_mpc

                # extra: save physical no-h units, used for emulator
                x = r
                y = profile_stacked
                data5 = np.array([x,y]).transpose()
                np.savetxt(f'{fname}_{ilam}.dat', data5, fmt='%-12g', header='rp [pMpc], DS [Msun/pc^2]')


                
                if plotting == True:
                    import matplotlib.pyplot as plt
                    plt.subplot(sb.nz, sb.nlam, iz*sb.nlam+ilam+1)

                    plt.plot(r, profile_select_mean/profile_stacked)#, label='%s'%(selection.replace('_','-')))
                    plt.legend()

                    plt.axhline(1,c='gray',ls='--')
                    plt.ylim(0.9, 1.4)
                    plt.xscale('log')
                    plt.xlabel(r'$\rm r_p\ [pMpc]$')
                
    
                ratio_list = profile_select_mean/profile_stacked
                outfile.write('%i %i '%(iz, ilam))
                for ratio in ratio_list:
                    outfile.write('%g '%(ratio))
                outfile.write('\n')
        outfile.close()
        #plt.savefig('test.png')
        #plt.show()
        '''

if __name__ == "__main__":
    sp = StackingProfiles(zmin=0.2, zmax=0.35)
    sp.read_halos()
    sp.stacking_profiles()


    # for sat_selection in ['Lcen2']:#,'Lcen2','Lcen4']:#
    #     for method in ['weighted']:#'matched', , 'shuffled'
    #         for selection in ['lum','lum_pmem','pmem0.9']:#, True]:, , 
    #             for i_run in [3]:#,4]:#, 4]:
    #                 for a_run in ['a']:#,'b','c','d','e','f']:
    #                     sp = StackingProfiles(i_run=i_run, a_run=a_run, obs_name='Sigma', sat_selection=sat_selection, method=method, selection=selection)
    #                     sp.read_halos()
    #                     sp.run_all_bins(use_thresholds=False)
                # if use_thresholds==False:
                #     plt.savefig('../../plots/stacking_profiles_comparison/bias_%s_%s_bins.pdf'%(method, sp.obs_name))
                #     plt.savefig('../../plots/stacking_profiles_%s/stacked_profile_bias_%s_bins.pdf'%(method, sp.obs_name))
                # if use_thresholds==True:
                #     plt.savefig('../../plots/stacking_profiles_comparison/bias_%s_%s_thresholds.pdf'%(method, sp.obs_name))
                # if selection==False:
                #     plt.savefig('../../plots/lcen/%s/stacked_profile_bias_%s_%s.pdf'%(method, sp.obs_name, sp.selection))
                # if selection==True:
    #plt.savefig('../../plots/lcen/%s/stacked_profile_bias_comparing_methods.pdf'%(method))

    #plt.show()