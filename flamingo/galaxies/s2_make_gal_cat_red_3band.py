#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
#import h5py
import fitsio
import pandas as pd
import os, sys

redshift = 0.5 #0.4 #0.4 #0.3

sim_name = 'L1000N3600/HYDRO_FIDUCIAL'
halo_finder = 'HBT' #'VR'
sim_label = sim_name.replace('/','_')
plot_loc = f'../../plots/galaxies/{sim_label}/z{redshift}/'
if os.path.isdir(plot_loc) == False: 
    os.makedirs(plot_loc)

chi_cut = 6 #2  #default is 6 
Mstar_lim = 2e+10
master_loc = f'/cosma8/data/do012/dc-wu5/cylinder/output_{sim_name}/z{redshift}/'
input_loc = master_loc + f'/model_mstar{Mstar_lim:.0e}_{halo_finder}/'
data, header = fitsio.read(f'{input_loc}/gals.fit', header=True)
#print(header)
mass_host = data['mass_host']
M_g = data['M_g'].flatten()
M_r = data['M_r'].flatten()
M_i = data['M_i'].flatten()
M_z = data['M_z'].flatten()
gr = M_g - M_r
ri = M_r - M_i
iz = M_i - M_z

color_list = np.array([gr, ri, iz])
color_name_list = ['g_r', 'r_i', 'i_z']

#### correlation coefficient
df = pd.DataFrame({'gr': gr, 'ri': ri, 'iz': iz})
correlation_matrix = df.corr()
print(correlation_matrix)
Cov = df.cov()
Cov = np.array(Cov)
iCov = np.linalg.inv(Cov)

#### fit the red-sequence with a straight line ####
plt.figure(figsize=(4,6))

Mvir_min = 5e14
sel = (mass_host > Mvir_min)

slope_list = []
intercept_list = []
std_list = []

for ic in range(3):
    color_name = color_name_list[ic]
    color = color_list[ic]
    x = M_z[sel] 
    y = color[sel]
    #plt.scatter(x,y,alpha=0.1)
    m, b = np.polyfit(x, y, 1) # slope, intercept    

    res = np.std(y - (m * x + b))
    slope_list.append(m)
    intercept_list.append(b)
    std_list.append(res)

    # plot the best-fit line
    x_plot = np.linspace(min(x), max(x))
    y_plot = m * x_plot + b
    
    ####
    plt.subplot(3,1,1+ic)
    plt.scatter(x, y, s=0.05, c='gray')
    line = plt.plot(x_plot, y_plot)#, label='lin regr')
    plt.fill_between(x_plot, y_plot-res, y_plot+res, interpolate=True, facecolor=line[0].get_c(), alpha=0.5)   
    plt.ylabel(color_name.replace('_','-'))
    #plt.ylim(0, 1)
    if ic==0: 
        #plt.legend()
        plt.title(sim_name.replace('_','-')+r', $\rm M_{vir}>%.e~h^{-1}M_\odot$'%Mvir_min, fontsize=10)
    if ic==2: plt.xlabel(r'$\rm M_z$')

plt.savefig(plot_loc+f'CMD_{Mvir_min:.0e}.png')



#### Calculate the chi^2 ####
slope_list = np.array(slope_list)
intercept_list = np.array(intercept_list)

Ngal = len(M_z.flatten())
chisq = np.zeros(Ngal)
for i in range(Ngal):
    color_exp = M_z[i] * slope_list + intercept_list
    diff = np.atleast_2d(color_list[:,i] - color_exp)
    x = diff @ iCov @ diff.T
    chisq[i] = x[0,0] # there must be a way to avoid for-loop. 


plt.figure()
dM = 0.5
for M_z_lim in np.arange(-25, -22, dM):
    sel_mag = (M_z > M_z_lim) & (M_z < M_z_lim + dM)
    #plt.hist(np.log10(chisq[sel_mag]), density=True)
    #### log x-axis ####
    #plt.subplot(121)
    Qlist = np.log10(chisq[sel_mag])
    nbins = 20
    if len(Qlist) == 0:
        pass
    else:
        lowest = min(Qlist)
        highest = max(Qlist)
        binsize = (highest-lowest)/(nbins*1.)
        x_list = np.zeros(nbins); y_list = np.zeros(nbins)
        for jj in np.arange(nbins):
            lower = lowest + binsize*jj
            upper = lowest + binsize*(jj+1)
            hh = len(Qlist[(Qlist>=lower)*(Qlist<upper)])
            #plt.scatter((lowest+(jj+0.5)*binsize), hh/(1.*len(Qlist)))
            x_list[jj] = (lowest+(jj+0.5)*binsize)
            y_list[jj] = hh/(1.*len(Qlist))
        norm = np.sum(y_list[x_list < np.log10(10)]) * binsize
        #norm = max(y_list)
        plt.plot(x_list, y_list/norm, 'o-', label=r'$%g<M_z<%g$'%(M_z_lim, M_z_lim+dM))


from scipy.stats import chi2
dof = 2
chi2_rand_numbers = chi2.rvs(dof, size=100000)
hist_data = plt.hist(np.log10(chi2_rand_numbers), fc='gray',
                     bins=50, density=True, alpha=0.5, label=r'$\chi^2_{\nu=%g}$'%dof)

plt.axvline(np.log10(chi_cut), c='k', ls='--', label=r'$\chi^2=%g$'%chi_cut)
plt.xlabel(r'$\rm \log_{10}\chi^2$')
plt.ylabel(r'$\rm PDF$')
plt.legend(loc=2)
plt.savefig(plot_loc+f'chi2_{chi_cut}.png')



#### find out density vs magnitude cut
from get_flamingo_info import get_flamingo_cosmo
cosmo = get_flamingo_cosmo(sim_name)
h = cosmo['h']
vol = (1000 * h)**3

M_z_cut_list = np.arange(-25, -18, 0.1)
den_list = []
for M_z_cut in M_z_cut_list:
    sel = (chisq < chi_cut)&(M_z < M_z_cut)
    den = len(chisq[sel]) / vol
    den_list.append(den)
den_list = np.array(den_list)
print('min max den_list', min(den_list), max(den_list))
from scipy.interpolate import interp1d
den_to_M = interp1d(den_list[::-1], M_z_cut_list[::-1])

##### Save the galaxy file #####
import os

def make_catalog(den_wanted):
    M_z_cut = den_to_M(den_wanted)
    
    sel = (chisq < chi_cut)&(M_z < M_z_cut)
    den = len(chisq[sel]) / vol
    print(den, f'{den:.0e}')
    output_loc = master_loc + f'model_redmagic_chi{chi_cut}_{den:.0e}_{halo_finder}/'
    if os.path.isdir(output_loc) == False: 
        os.makedirs(output_loc)
    
    data, header = fitsio.read(f'{input_loc}/gals.fit', header=True)
    
    from astropy.io import fits
    
    cols=[
      #fits.Column(name='hid_host', format='K', array=data['hid_host'][sel]),
      #fits.Column(name='hid_sub', format='K', array=data['hid_sub'][sel]),
      fits.Column(name='mass_host', unit='Mvir, Msun/h', format='E', array=data['mass_host'][sel]),
      #fits.Column(name='mass_sub', unit='Mvir, Msun/h', format='E', array=data['mass_sub'][sel]),
      fits.Column(name='px', unit='Mpc/h', format='D', array=data['px'][sel]),
      fits.Column(name='py', unit='Mpc/h', format='D', array=data['py'][sel]),
      fits.Column(name='pz', unit='Mpc/h', format='D', array=data['pz'][sel]),
      fits.Column(name='vx', format='D', array=data['vx'][sel]),
      fits.Column(name='vy', format='D', array=data['vy'][sel]),
      fits.Column(name='vz', format='D', array=data['vz'][sel]),
      #### Magnitude information ####
      fits.Column(name='M_g', unit='', format='D', array=data['M_g'][sel]),
      fits.Column(name='M_r', unit='', format='D', array=data['M_r'][sel]),
      fits.Column(name='M_i', unit='', format='D', array=data['M_i'][sel]),
      fits.Column(name='M_z', unit='', format='D', array=data['M_z'][sel]),
      #fits.Column(name='iscen', unit='', format='K',array=data['iscen'][sel]),
    ]
    coldefs = fits.ColDefs(cols)
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    fname = 'gals.fit'
    tbhdu.writeto(f'{output_loc}/{fname}', overwrite=True)

for den_wanted in [1e-3, 6e-3, 1e-2]: 
    make_catalog(den_wanted)


