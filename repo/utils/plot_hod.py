#!/usr/bin/env python
import numpy as np
from scipy import spatial
rmax_tree = 4
import yaml
import fitsio
import sys

class PlotHOD(object):
    def __init__(self, yml_fname):
        with open(yml_fname, 'r') as stream:
            try:
                self.para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.redshift = self.para['redshift']
        self.mdef = self.para['mdef']
        self.Om0 = self.para['OmegaM']

        output_loc = self.para['output_loc']
        model_name = self.para['model_name']
        self.out_path = f'{output_loc}/model_{model_name}/'
        self.ofname = f'{self.out_path}/hod_recovered.dat'

    def calc_hod(self, Mmin):
        if self.para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            readcat = ReadMiniUchuu(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'uchuu':
            from read_uchuu import ReadUchuu
            readcat = ReadUchuu(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'abacus_summit':
            from read_abacus_summit import ReadAbacusSummit
            readcat = ReadAbacusSummit(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'flamingo':
            from read_flamingo import ReadFlamingo
            readcat = ReadFlamingo(self.para['nbody_loc'], self.redshift)

        if self.para['nbody'] == 'tng_dmo':
            from read_tng_dmo import ReadTNGDMO
            halofinder = para.get('halofinder', 'rockstar')
            readcat = ReadTNGDMO(self.para['nbody_loc'], halofinder, self.redshift)
            print('halofinder', halofinder)

       
        readcat.read_halos(Mmin)
        mass = readcat.mass
        x_halo = readcat.xh
        y_halo = readcat.yh
        z_halo = readcat.zh

        gal_fname = f'{self.out_path}/gals.fit'
        data, header = fitsio.read(gal_fname, header=True)

        x_gal = data['px']
        y_gal = data['py']
        z_gal = data['pz']

        gal_position = np.dstack([x_gal, y_gal, z_gal])[0]
        gal_tree = spatial.cKDTree(gal_position)

        halo_position = np.dstack([x_halo, y_halo, z_halo])[0]
        halo_tree = spatial.cKDTree(halo_position)
        indexes_tree = halo_tree.query_ball_tree(gal_tree, r=rmax_tree)

        rhocrit = 2.775e11 # h-unit
        if self.mdef == '200m':
            radius = (3 * mass / (4 * np.pi * rhocrit * self.Om0 * 200.))**(1./3.) # cMpc/h (chimp)
        elif self.mdef == 'vir':
            OmegaM_z = self.Om0 * (1+ self.redshift)**3 / (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)
            x = OmegaM_z - 1
            Delta_vir_c = 18 * np.pi**2 + 82 * x - 39 * x**2
            rhocrit_z = rhocrit * (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)/(1+self.redshift)**3 # gotcha!
            radius = (3 * mass / (4 * np.pi * rhocrit_z * Delta_vir_c))**(1./3.)

        nhalo = len(x_halo)
        ngal = []
        for i_halo in range(nhalo):
            gal_ind = indexes_tree[i_halo]
            x_cen = x_halo[i_halo]
            y_cen = y_halo[i_halo]
            z_cen = z_halo[i_halo]
            #r = (x_gal[gal_ind] - x_cen)**2 + (y_gal[gal_ind] - y_cen)**2 + (z_gal[gal_ind] - z_cen)**2 
            #r = np.sqrt(r)
            #ngal.append(len(r[r <= radius[i_halo]*(1+1e-4)]))
            indx = gal_tree.query_ball_point([x_cen, y_cen, z_cen], radius[i_halo])
            ngal.append(len(indx))
        ngal = np.array(ngal)

        x = np.log(mass)
        y = ngal

        nbins_per_decade = 10
        n_decade = (np.log10(max(mass))-np.log10(min(mass)))
        nbins = int(nbins_per_decade*n_decade + 1e-4) 
        x_bins = np.linspace(min(x), max(x), nbins+1)
        x_bin_mean = []
        y_bin_mean = []
        y_bin_std = []
        for i in range(nbins):
            sel = (x > x_bins[i])&(x < x_bins[i+1])
            x_bin_mean.append(np.mean(x[sel]))
            y_bin_mean.append(np.mean(y[sel]))
            y_bin_std.append(np.std(y[sel]))

        y_bin_mean = np.array(y_bin_mean)
        y_bin_std = np.array(y_bin_std)

        

        data = np.array([np.exp(x_bin_mean), y_bin_mean, y_bin_std]).transpose()
        np.savetxt(self.ofname, data, fmt='%-12g', header='mass, mean(Ngal), std(N_gal)')

        return np.exp(x_bin_mean), y_bin_mean, y_bin_std


    def plot_hod(self):
        m, mean_Ngal, std_ngal = np.loadtxt(self.ofname, unpack=True)
        return m, mean_Ngal, std_ngal

if __name__ == "__main__":
    yml_fname = sys.argv[1] 
    #./plot_hod.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml

    ch = PlotHOD(yml_fname)
    mass, hod_mean, hod_std = ch.calc_hod(Mmin=1e11)

    
    # use plot_hod_mor.ipynb for plotting
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    plt.loglog(mass, hod_mean, 'o-')

    plt.subplot(122)
    plt.semilogx(mass, hod_std)

    plt.show()