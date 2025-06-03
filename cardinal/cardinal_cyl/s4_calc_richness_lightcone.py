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


### TODO: add percolation, across boundaries
### TODO: add pmem weights # Done
### TODO: add R_lam iteration # Done
### TODO: background subtraction # Done

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
#plt.scatter(ra_all[::100], dec_all[::100], c='C0', s=0.01)


# read in halos (ra, dec, chi), sorted by mass
data, header = fitsio.read(output_loc+'/../halos_from_gold.fit', header=True)
#print(header)
M_cen_all = data['mvir']
ids_cen_all = data['haloid']
ra_cen_all = data['ra']
dec_cen_all = data['dec']
chi_cen_all = data['chi']

# plt.scatter(ra_cen_all[::100], dec_cen_all[::100], c='C1', s=1)
# plt.savefig('test.png')
# exit()
# calculate richness for one redshift bin
class CylinderRichnessLightcone(object):
    def __init__(self, zmin, zmax):#, sample, method, radius, depth):
        self.zmin = zmin
        self.zmax = zmax
        self.zmid = 0.5 * (self.zmin + self.zmax)
        self.save_name = 'z_%g_%g'%(self.zmin, self.zmax)

        self.richness_file = output_loc+'temp/Ncyl_%s.dat'%(self.save_name)
        self.chi_min = cosmo.comoving_distance(zmin).value
        self.chi_max = cosmo.comoving_distance(zmax).value
        #print('chi range', self.chi_min, self.chi_max)

        self.density_bg = density_bg_inerp(self.zmid)

    def get_galaxies_ra_dec(self): # for cone
        sel = (chi_all >= self.chi_min - 5*depth)&(chi_all <= self.chi_max + 5*depth) # not working for gauss
        self.chi_gal = chi_all[sel]
        self.ra_gal = ra_all[sel] # deg
        self.dec_gal = dec_all[sel] # deg
        print('number of galaxies', len(self.chi_gal))

    def get_richness_cone(self, ra_cen, dec_cen, chi_cen): 
        if which_pmem == 'quad':
            sel1 = (np.abs(self.chi_gal - chi_cen) < depth) # not working for gauss
        #Ncyl = len(self.chi_gal)
        #ang_sep_2 = np.zeros(Ncyl) + 1e5 # same lengh as self.chi_gal
        dz = np.abs(self.chi_gal[sel1] - chi_cen)
        d_ra = self.ra_gal[sel1] - ra_cen
        d_dec = self.dec_gal[sel1] - dec_cen
        ang_sep_2 = d_ra**2 * np.cos(dec_cen*np.pi/180.)**2 + d_dec**2
        r = np.sqrt(ang_sep_2) * (np.pi/180.) * chi_cen # projected separation in Mpc/h

        #radius = 1 # pMpc/h
        #sel_uni = (r<radius)*(dz<300)
        #print('len(r[sel_uni]) = ', len(r[sel_uni]))
        #exit()

        # if use_rlambda == False:
        #     # if self.unit == 'chimp':
        #     #     ang_cyl = radius / chi_cen * (180. / np.pi) # deg
        #     # if self.unit == 'phys':
        # DA = chi_cen / (1+self.zmid)  # pMpc/h
        # ang_cyl = (radius / DA) * (180. / np.pi) # deg
        # sel = (ang_sep_2 < ang_cyl**2)
        # print('len(ang_sep_2[sel])', len(ang_sep_2[sel]))
        # exit()

        #     #if weight == 'None':
        #     richness = len(ang_sep_2[sel])
        redshift = z_chi(chi_cen)
        #print('redshift', redshift)
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
        # if perc == True:
        #     self.chi_gal[sel] = -1  # effectively removing these galaxies # can't do arr[sel1][sel2]=-1
        #     #print('after', len(self.chi_gal[self.chi_gal > 0]))
        #return richness
        return rlam, Ncyl, redshift

    def measure_richness(self):
        sel = (chi_cen_all > self.chi_min)&(chi_cen_all < self.chi_max)&(M_cen_all > 1e14)
        M_cen = M_cen_all[sel]
        ids_cen = ids_cen_all[sel]
        ra_cen = ra_cen_all[sel]
        dec_cen = dec_cen_all[sel]
        chi_cen = chi_cen_all[sel]

        # sort by mass
        ind = np.argsort(-M_cen)
        M_cen = M_cen[ind]
        ids_cen = ids_cen[ind]
        ra_cen = ra_cen[ind]
        dec_cen = dec_cen[ind]
        chi_cen = chi_cen[ind]


        # px_cen = data['px'][sel]
        # py_cen = data['py'][sel]
        # pz_cen = data['pz'][sel]

        #z_cos = z_cos[sel]
        #M_cen = M_cen[sel]
        #chi_cen = cosmo.comoving_distance(z_cos).value
        # px_cen = chi_cen * np.sin(dec_cen * np.pi / 180.)
        # py_cen = chi_cen * np.cos(dec_cen * np.pi / 180.) * np.cos(ra_cen * np.pi / 180.)
        # pz_cen = chi_cen * np.cos(dec_cen * np.pi / 180.) * np.sin(ra_cen * np.pi / 180.)
        #px_cen, py_cen, pz_cen = astropy.coordinates.spherical_to_cartesian(chi_cen, dec_cen*np.pi/180., ra_cen*np.pi/180.)
        # px_cen = px_cen.value
        # py_cen = py_cen.value
        # pz_cen = pz_cen.value
        #Radius_cen = 1 + np.zeros(len(ids_cen)) #data['r_lambda'][sel]

        print('number of halos', len(ids_cen))

        self.get_galaxies_ra_dec()

        ## looping through halos
        outfile2 = open(self.richness_file,'w')
        outfile2.write('id Mvir Ncyl Rcyl z_cos\n')
        id_to_start = 0
        for i in range(id_to_start, len(ids_cen)):
            #radius = Radius_cen[i]
            ## calculation richness
            rlam, richness, redshift = self.get_richness_cone(ra_cen[i], dec_cen[i], chi_cen[i])
            outfile2.write('%-12i \t'%(ids_cen[i])) # write ID first
            outfile2.write('%-12e \t'%(M_cen[i])) # need mass. ID is not unique
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

