#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py

import matplotlib as mpl
#mpl.rcParams['font.size'] = 16
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif'] # not working
mpl.rcParams['axes.labelsize'] = 25
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['axes.titlesize'] = 25 # this needs to be a bigger number



#plt.rcParams["font.family"] = "Times New Roman"

out_loc = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-0/z0p3'

fname1 = out_loc+'/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3.richness_d30.hdf5'

fname2 = out_loc+'/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3_noperc/memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3_noperc.richness_d30.hdf5'


f = h5py.File(fname1,'r')
halos = f['halos']
print(halos.dtype)
mass1 = halos['mass']
lam1 = halos['lambda']


f = h5py.File(fname2,'r')
halos = f['halos']
mass2 = halos['mass']
lam2 = halos['lambda']


print(np.max(abs(mass1 - mass2)))

sel = (mass1 < 10**12.6)
#plt.scatter(lam1[sel], lam2[sel])
plt.hist(lam1[sel], density = True, label='perc', alpha=0.5)
plt.hist(lam2[sel], density = True, label='no perc', alpha=0.5)
plt.legend()
plt.xlabel(r'cylinder richness')
plt.yscale('log')
plt.title(r'$\rm log_{10}M = 12.5 - 12.6$')
plt.savefig('../../plots/richness/perc_lambda_hist.png')

plt.figure()
sel1 = (lam1 > 20)&(lam1 < 30)
plt.hist(np.log10(mass1)[sel1], density = True, label='perc', alpha=0.5)
sel2 = (lam2 > 20)&(lam2 < 30)
plt.hist(np.log10(mass2)[sel2], density = True, label='no perc', alpha=0.5)
plt.legend()
plt.xlabel(r'cylinder richness')
plt.yscale('log')
plt.title(r'$\rm 20 < \lambda < 30$')
plt.savefig('../../plots/richness/perc_mass_hist.png')




#plt.show()



'''
#fname1 = out_loc+'Sunayama_cylinder_d30_noperc/richness_100_200_M1e+14_test.dat'
#fname2 = out_loc+'Sunayama_cylinder_d30_back/richness_100_200_M1e+14_test.dat'

# first check if the gids are unique

#gid, mass, px, py, pz, rlam, lam 
data1 = np.loadtxt(fname1)
gid1 = data1[:46196,0]
print('check uniqueness', len(gid1), len(np.unique(gid1)))

lam1 = data1[:46196,6]
lgm = np.log10(data1[:46196,1])

data2 = np.loadtxt(fname2)
lam2 = data2[:46196,6]

plt.figure(figsize=(21,7))
plt.subplot(131)
plt.scatter(lgm, np.log10(lam1), c='C0', alpha=0.2)
plt.xlabel(r'$\rm log_{10}M$')
plt.ylabel(r'$\rm log_{10}\lambda$')
plt.title('no perc')

plt.subplot(132)

print(len(lgm), len(np.log10(lam2)))

plt.scatter(lgm, np.log10(lam2), c='C1', alpha=0.2)
plt.xlabel(r'$\rm log_{10}M$')
plt.ylabel(r'$\rm log_{10}\lambda$')
plt.title('perc')

diff = (lam1 - lam2)/lam1

plt.subplot(133)
plt.scatter(lgm, diff)
plt.xlabel(r'$\rm log_{10}M$')
plt.ylabel(r'$\rm (\lambda_{no\ perc} - \lambda_{perc})/\lambda_{no\ perc}$')

plt.savefig('scat.png')
#plt.show() # it will not show
'''