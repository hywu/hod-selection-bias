#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import fitsio
#import astropy.coordinates
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.286)
z_list = np.arange(0,1,0.01)
chi_list = cosmo.comoving_distance(z_list).value
from scipy.interpolate import interp1d
z_chi = interp1d(chi_list, z_list)
from sklearn.neighbors import BallTree

### uses percolation, across boundaries
### uses pmem weights
### uses R_lam iteration
### uses background subtraction

perc = True
use_rlambda = True
which_pmem = 'quad'
if which_pmem == 'quad':
    from hod.utils.pmem_weights_quad import pmem_weights_dchi, volume_dchi
dchi_max = 3000 # not necessary for lightcone code

chisq_cut = int(sys.argv[1])#100
output_loc = '/projects/hywu/cluster_sims/cluster_finding/data/cardinal_cyl/'
output_loc += f'model_chisq{chisq_cut}/'

depth = 180


z, den = np.loadtxt(output_loc+'gal_density.dat', unpack=True)
from scipy.interpolate import interp1d
density_bg_inerp = interp1d(z, den)

# read in galaxies (ra, dec, chi)
data, header = fitsio.read(output_loc+'gals.fit', header=True)
ra_all = data['ra']
dec_all = data['dec']
chi_all = data['chi']
'''
ra_all = np.zeros(1000)
dec_all = np.zeros(1000)
chi_all = np.zeros(1000)
'''

# read in halos (ra, dec, chi), sorted by mass
data, header = fitsio.read(output_loc+'/../halos_from_gold.fit', header=True)
M_cen_all = data['mvir']
ids_cen_all = data['haloid']
ra_cen_all = data['ra']
dec_cen_all = data['dec']
chi_cen_all = data['chi']


# calculate richness for one redshift bin
class CylinderRichnessLightcone(object):
    def __init__(self, zmin, zmax):
        self.zmin = zmin
        self.zmax = zmax
        self.zmid = 0.5 * (self.zmin + self.zmax)
        self.save_name = 'z_%g_%g'%(self.zmin, self.zmax)

        self.richness_file = output_loc+'temp/Ncyl_%s.dat'%(self.save_name)
        self.chi_min = cosmo.comoving_distance(zmin).value
        self.chi_max = cosmo.comoving_distance(zmax).value
        #print('chi range', self.chi_min, self.chi_max)

        self.density_bg = density_bg_inerp(self.zmid)



        #### get galaxies
        if which_pmem == 'quad':
            sel = (chi_all >= self.chi_min - depth)&(chi_all <= self.chi_max + depth) 
        if which_pmem == 'gauss':
            sel = (chi_all >= self.chi_min - 5 * depth)&(chi_all <= self.chi_max + 5 * depth) 
            
        self.chi_gal = chi_all[sel]
        self.ra_gal = ra_all[sel] # deg
        self.dec_gal = dec_all[sel] # deg
        self.gal_taken = np.zeros(len(self.chi_gal)) # for percolation



        #### get halos ####
        sel_halo = (chi_cen_all > self.chi_min - depth)&(chi_cen_all < self.chi_max + depth) # include extra dpeth for percolation
        sel_halo = sel_halo & (M_cen_all > 1e13)
        M_cen = M_cen_all[sel_halo]
        ids_cen = ids_cen_all[sel_halo]
        ra_cen = ra_cen_all[sel_halo]
        dec_cen = dec_cen_all[sel_halo]
        chi_cen = chi_cen_all[sel_halo]

        # sort by mass
        ind = np.argsort(-M_cen)
        self.ids_cen = ids_cen[ind]
        self.M_cen = M_cen[ind]
        self.ra_cen = ra_cen[ind]
        self.dec_cen = dec_cen[ind]
        self.chi_cen = chi_cen[ind]
        self.ncen = len(self.M_cen)
        print('number of halos', len(ids_cen))


        ##### build a tree! for a rough radius cut
        # haversine takes: latitude (dec), longitude (RA)
        X_gal = np.column_stack([self.dec_gal*np.pi/180., self.ra_gal*np.pi/180.])
        tree = BallTree(X_gal, metric='haversine')
        print('number of galaxies', len(self.chi_gal))
        X_cen = np.column_stack([self.dec_cen*np.pi/180., self.ra_cen*np.pi/180.])
        theta_max = 2 / self.chi_min # set max to 2 hiMpc.
        self.indexes_tree = tree.query_radius(X_cen, r=theta_max)




    def get_richness_cone(self, i_cen):#, ra_cen, dec_cen, chi_cen): # one center at a time
        gal_ind = self.indexes_tree[i_cen]

        ra_gal = self.ra_gal[gal_ind]
        dec_gal = self.dec_gal[gal_ind]
        chi_gal = self.chi_gal[gal_ind]

        ra_cen = self.ra_cen[i_cen]
        dec_cen = self.dec_cen[i_cen]
        chi_cen = self.chi_cen[i_cen]


        #### step 1: cut the LOS ####
        if which_pmem == 'quad':
            sel_z = (np.abs(chi_gal - chi_cen) < depth)
        if which_pmem == 'quad':
            sel_z = (np.abs(chi_gal - chi_cen) < 5 * depth)

        sel_z = sel_z & (self.gal_taken[gal_ind] < 0.8) # percolation threshold = 0.8

        dz = np.abs(chi_gal[sel_z] - chi_cen)
        # calculate small distances using brute force
        d_ra = ra_gal[sel_z] - ra_cen
        d_dec = dec_gal[sel_z] - dec_cen
        ang_sep_2 = d_ra**2 * np.cos(dec_cen*np.pi/180.)**2 + d_dec**2
        r = np.sqrt(ang_sep_2) * (np.pi/180.) * chi_cen # projected separation in Mpc/h
        redshift = z_chi(chi_cen)
        scale_factor = 1./(1.+redshift)

        if use_rlambda == True:
            rlam_ini = 1
            rlam = rlam_ini
            for iteration in range(100):
                pmem = pmem_weights_dchi(dz, r/rlam, dchi_max=dchi_max, depth=depth)
                
                Ncyl = np.sum(pmem)

                nbg = self.density_bg * volume_dchi(rlam, depth)
                Ncyl = max(Ncyl - nbg, 0) # can't be negative

                #else:
                #    print('BUG!!')

                rlam_old = rlam
                rlam = (Ncyl/100.)**0.2 / scale_factor # phys -> comoving
                if abs(rlam - rlam_old) < 1e-5 or rlam < 1e-6:
                    break

        # if weight == 'C19':
        #     sigma_chi = 300
        #     w = 1 - (self.chi_gal[sel] - chi_cen)**2 / sigma_chi**2
        #     richness = np.sum(w[w > 0])

        #print('before', len(self.chi_gal[self.chi_gal > 0]))
        if perc == True:
            sel_mem = (r < rlam)
            self.gal_taken[np.array(gal_ind)[sel_z][sel_mem]] += pmem[sel_mem]

            #ind_taken = np.arange(len(self.gal_taken))[sel_z][sel_mem]
            #self.gal_taken[ind_taken] += pmem[sel_mem]
            #self.chi_gal[sel] = -1  # effectively removing these galaxies # can't do arr[sel_z][sel2]=-1
            #print('after', len(self.chi_gal[self.chi_gal > 0]))
        #return richness
        return rlam, Ncyl, redshift

    def measure_richness(self):

        ## looping through halos
        outfile2 = open(self.richness_file,'w')
        outfile2.write('id Mvir Ncyl Rcyl z_cos\n')
        #id_to_start = 0
        for i in range(self.ncen):#range(id_to_start, len(ids_cen)):
            #radius = Radius_cen[i]
            ## calculation richness
            rlam, richness, redshift = self.get_richness_cone(i)#ra_cen[i], dec_cen[i], chi_cen[i])
            if redshift > self.zmin and redshift < self.zmax: # excluding extra percolating halos
                outfile2.write('%-12i \t'%(self.ids_cen[i])) # write ID first
                outfile2.write('%-12e \t'%(self.M_cen[i])) # need mass. ID is not unique
                outfile2.write('%-12g \t'%(richness))
                outfile2.write('%-12g '%(rlam))
                outfile2.write('%-12g '%(redshift))
                outfile2.write('\n')

        outfile2.close()

dz = 0.01
zmin_list = np.arange(0.2, 0.641, dz)

def calc_one_bin(ibin):
    if ibin < len(zmin_list):
        zmin = zmin_list[ibin]
        zmax = zmin + dz
        crc = CylinderRichnessLightcone(zmin=zmin, zmax=zmin+0.01)
        crc.measure_richness()

if __name__ == "__main__":
    #calc_one_bin(0)
    #exit()

    n_parallel = len(zmin_list)
    
    import os
    from concurrent.futures import ProcessPoolExecutor
    n_cpu = os.getenv('SLURM_CPUS_PER_TASK')

    if n_cpu is not None:
        n_cpu = int(n_cpu)
        print(f'Assigned CPUs: {n_cpu}') 
    else:
        print('Not running under SLURM or the variable is not set.') 
        n_cpu = 1

    n_workers = int(max(1, n_cpu*0.8))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(calc_one_bin, range(n_parallel)):
            if result: print(result)  # output error
    #stop = timeit.default_timer()
    
    
    from hod.utils.merge_files import merge_files

    merge_files(in_fname=f'{output_loc}/temp/Ncyl_*.dat', 
        out_fname=f'{output_loc}/Ncyl.fit', 
        nfiles_expected=n_parallel, delete=False)

