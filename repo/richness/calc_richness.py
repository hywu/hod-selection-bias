#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os, sys


#### things can be moved to yml files ####
# sample = 'fid'
# method = 'cylinder'
boxsize = 1100
perc = True
radius = 'rlambda'

#loc = '/users/PAS0023/osu0218/Project/For_Heidi/'
#loc = '/bsuhome/hwu/work/For_Heidi/'
# halo_fname = loc + 'RShalos-fid0cosmo2-z0p3.hdf5'
# gal_fname = loc + '%s_HOD_catalog.hdf5'%(sample)

phase = sys.argv[1]
run_name = sys.argv[2] #'memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3'
depth = int(sys.argv[3]) #60

loc = f'/bsuhome/hwu/scratch/Projection_Effects/Catalogs/fiducial-{phase}/z0p3/'




halo_fname = loc + f'RShalos-fid{phase}cosmo2-z0p3.hdf5'
gal_fname = loc + run_name + '.hdf5'

out_loc = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/{run_name}'
if perc == False:
    out_loc += '_noperc'

print(loc)
print(out_loc)

Mmin = 10**12.5




# if np.isclose(Mmin, 10**12.5):
#     out_loc += 'M_{Mmin:.2e}_{Mmax:.2e}'


if os.path.isdir(out_loc)==False:
    os.makedirs(out_loc)

############################################


#run_name = f'{sample}_{method}_d{depth}'



# if os.path.isdir(out_loc)==False:
#     os.makedirs(out_loc)

#### read in halos ####
f = h5py.File(halo_fname,'r')
halos = f['halos']
#print(halos.dtype)
mass = halos['mass'] # m200b
sel = (mass > Mmin)
x_halo = halos['x'][sel]
y_halo = halos['y'][sel]
z_halo = halos['z'][sel]
rvir = halos['rvir'][sel]/1e3
mass = mass[sel]
gid = halos['gid'][sel] # use gid as halo id

index = np.argsort(-mass)
x_halo = x_halo[index]
y_halo = y_halo[index]
z_halo = z_halo[index]
mass = mass[index]
gid = gid[index]
print('finished halos')


#### read in galaxies ####
f = h5py.File(gal_fname,'r')
particles  = f['particles']
#print(particles.dtype)
x_gal = particles['x']
y_gal = particles['y']
z_gal = particles['z']
print('finished galaxies')

'''
#### periodic boundary conditions ####  TODO

pad = 10
sel = (x > -pad)&(x < boxsize - pad)&(y < -pad)&(z < boxsize - pad)
x_all = x_all[sel]
y_all = y_all[sel]
z_all = z_all[sel]


sel = (z_gal < depth) 
x_gal_append1 = x_gal[sel]
y_gal_append1 = y_gal[sel]
z_gal_append1 = z_gal[sel] + boxsize

sel = (z_gal > boxsize - depth) 
x_gal_append2 = x_gal[sel]
y_gal_append2 = y_gal[sel]
z_gal_append2 = z_gal[sel] - boxsize
'''


class CalcRichness(object):
    def __init__(self, pz_min, pz_max):
        self.pz_min = pz_min
        self.pz_max = pz_max
        d_pz = pz_max - pz_min

        # periodic boundary condition in pz direction
        if pz_min < depth: # near the lower boundary
            sel_pz1 = (z_gal < pz_max + 1.2*depth) 
            sel_pz2 = (z_gal > boxsize - 1.2*depth)
            x_gal_slice = np.append(x_gal[sel_pz1], x_gal[sel_pz2]) # oh well...
            y_gal_slice = np.append(y_gal[sel_pz1], y_gal[sel_pz2])
            z_gal_slice = np.append(z_gal[sel_pz1], z_gal[sel_pz2] - boxsize)

            # for halos: padding for percolation
            sel_pz1 = (z_halo < pz_max + 1.2*depth) 
            sel_pz2 = (z_halo > boxsize - 1.2*depth)
            self.x_halo_padded = np.append(x_halo[sel_pz1], x_halo[sel_pz2]) # oh well...
            self.y_halo_padded = np.append(y_halo[sel_pz1], y_halo[sel_pz2])
            self.z_halo_padded = np.append(z_halo[sel_pz1], z_halo[sel_pz2] - boxsize)
            self.mass_padded = np.append(mass[sel_pz1], mass[sel_pz2])
            self.gid_padded = np.append(gid[sel_pz1], gid[sel_pz2])


        elif pz_max > boxsize - depth: # near the upper boundary
            sel_pz1 = (z_gal > pz_min - 1.2*depth)
            sel_pz2 = (z_gal < 1.2*depth) 
            x_gal_slice = np.append(x_gal[sel_pz1], x_gal[sel_pz2])
            y_gal_slice = np.append(y_gal[sel_pz1], y_gal[sel_pz2])
            z_gal_slice = np.append(z_gal[sel_pz1], z_gal[sel_pz2] + boxsize)

            # for halos
            sel_pz1 = (z_halo > pz_min - 1.2*depth)
            sel_pz2 = (z_halo < 1.2*depth) 
            self.x_halo_padded = np.append(x_halo[sel_pz1], x_halo[sel_pz2])
            self.y_halo_padded = np.append(y_halo[sel_pz1], y_halo[sel_pz2])
            self.z_halo_padded = np.append(z_halo[sel_pz1], z_halo[sel_pz2] + boxsize)
            self.mass_padded = np.append(mass[sel_pz1], mass[sel_pz2])
            self.gid_padded = np.append(gid[sel_pz1], gid[sel_pz2])

        else: # safe in the middle
            sel_pz = (z_gal > pz_min - 1.2*depth)&(z_gal < pz_max + 1.2*depth)
            x_gal_slice = x_gal[sel_pz]
            y_gal_slice = y_gal[sel_pz]
            z_gal_slice = z_gal[sel_pz]

            # for halos
            sel_pz = (z_halo > pz_min - 1.2*depth)&(z_halo < pz_max + 1.2*depth)
            self.x_halo_padded = x_halo[sel_pz]
            self.y_halo_padded = y_halo[sel_pz]
            self.z_halo_padded = z_halo[sel_pz]
            self.mass_padded = mass[sel_pz]
            self.gid_padded = gid[sel_pz]

        # periodic boundary condition in x-y direction # only galaxies, no need to do this for halos
        x_all = []
        y_all = []
        z_all = []
        for x_pm in [-1, 0, 1]:
            for y_pm in [-1, 0, 1]:
                x_all.extend(x_gal_slice + x_pm * boxsize)
                y_all.extend(y_gal_slice + y_pm * boxsize)
                z_all.extend(z_gal_slice)

        x_all = np.array(x_all)
        y_all = np.array(y_all)
        z_all = np.array(z_all)
        padding = 10
        sel = (x_all > -padding)&(x_all < boxsize + padding)&(y_all > -padding)&(y_all < boxsize + padding)

        self.x_gal = x_all[sel]
        self.y_gal = y_all[sel]
        self.z_gal = z_all[sel]


    def plot_check_pbc(self):
        plt.figure(figsize=(21,7))
        plt.subplot(131, aspect='equal')
        plt.scatter(self.x_gal[::1000], self.y_gal[::1000])
        plt.axhline(0)
        plt.axhline(boxsize)
        plt.axvline(0)
        plt.axvline(boxsize)

        plt.subplot(132, aspect='equal')
        plt.scatter(self.y_gal[::1000], self.z_gal[::1000])
        plt.axvline(0)
        plt.axvline(boxsize)
        plt.axhline(self.pz_min)
        plt.axhline(self.pz_max)

        plt.subplot(133, aspect='equal')
        plt.scatter(self.x_gal[::1000], self.z_gal[::1000])
        plt.axvline(0)
        plt.axvline(boxsize)
        plt.axhline(self.pz_min)
        plt.axhline(self.pz_max)

        plt.savefig('pbc.png')





    def get_richness_cone(self, x_cen, y_cen, z_cen):
        r = (self.x_gal - x_cen)**2 + (self.y_gal - y_cen)**2 
        r = np.sqrt(r)
        d = np.abs(self.z_gal - z_cen)
        r_z = r[(r < 3)&(d < depth)]  # a generous cut before perc

        if radius == 'rlambda':
            rlam_ini = 1
            rlam = rlam_ini
            for iteration in range(100):
                ngal = len(r_z[(r_z < rlam)])
                rlam_old = rlam
                rlam = (ngal/100.)**0.2
                #print(rlam, rlam_old)
                if abs(rlam - rlam_old) < 1e-5:
                    break

        sel_mem = (r < rlam)&(d < depth) # a generous cut before perc
        lam = len(r[sel_mem])

        if perc == True:
            self.x_gal = self.x_gal[~sel_mem]
            self.y_gal = self.y_gal[~sel_mem]
            self.z_gal = self.z_gal[~sel_mem]
        #print('len(self.x_gal > 0)', len(self.x_gal > 0), rlam, lam)
        return rlam, lam

    def measure_richness(self):
        #sel_pz = (z_halo > self.pz_min)&(z_halo < self.pz_max)
        #self.x_halo_padded = x_halo[sel_pz]
        #self.y_halo_padded = y_halo[sel_pz]
        #self.z_halo_padded = z_halo[sel_pz]

        nh = len(self.x_halo_padded)
        print('nh =', nh)

        ofname = f'{out_loc}/richness_{self.pz_min}_{self.pz_max}_d{depth}.dat'

        if os.path.exists(ofname) == False:
            id_to_start = 0
        else:
            d = np.loadtxt(ofname)
            if len(d) == 0:
                id_to_start = 0
            else:
                id_to_start = len(d[:,0])
        print('id_to_start', id_to_start)

        for ih in range(id_to_start, nh):

            outfile = open(ofname, 'a')
            if ih == 0: outfile.write('#gid, mass, px, py, pz, rlam, lam \n')


            rlam, lam = self.get_richness_cone(self.x_halo_padded[ih], self.y_halo_padded[ih], self.z_halo_padded[ih]) # use padded halos

            if self.z_halo_padded[ih] > self.pz_min and self.z_halo_padded[ih] < self.pz_max: # discard padded halos

                outfile.write('%12i %15e %12g %12g %12g %12g %12g \n'%(self.gid_padded[ih], self.mass_padded[ih], self.x_halo_padded[ih], self.y_halo_padded[ih], self.z_halo_padded[ih], rlam, lam))
            outfile.close()


def calc_one_bin(ibin):
    cr = CalcRichness(pz_min=ibin*100, pz_max=(ibin+1)*100)
    cr.measure_richness()
    #if ibin < len(zmin_list):
    # zmin = zmin_list[ibin]
    # zmax = zmin + dz
    # crc = AlternativeRichness(zmin=zmin, zmax=zmin+0.01, sample=sample, method=method, radius=radius_input, depth=depth, chisq_cut=chisq_cut)
    # crc.measure_richness()


if __name__ == '__main__':
    # cr = CalcRichness(pz_min=100, pz_max=200)
    # cr.measure_richness()
    # for i in [0]:#range(12):
    #     cr = CalcRichness(pz_min=i*100, pz_max=(i+1)*100)
    #     #cr.plot_check_pbc()
    #     

    
    # parallel
    from multiprocessing import Pool
    #for i_round in range(n_round):
    n_job2 = 11
    p = Pool(n_job2)
    p.map(calc_one_bin, range(n_job2))
    






'''
def rlambda(r, rlam_ini):
    rlam = rlam_ini
    for iteration in range(10):
        sel = (r < rlam)
        ngal = len(r[sel])
        rlam_old = rlam
        rlam = (ngal/100.)**0.2
        #print(rlam, rlam_old)
        if abs(rlam - rlam_old) < 1e-5:
            break
    return rlam, len(r < rlam)
'''


# print(len(d[sel]))


# data = np.array([x_halo[ih], y_halo[ih], z_halo[ih]]).transpose()
# np.savetxt('halo.dat', data, fmt='%-12g')

# data = np.array([x_gal[sel], y_gal[sel], z_gal[sel]]).transpose()
# np.savetxt('gal.dat', data, fmt='%-12g')


# plt.hist(r[r<10])



# make HDF5 
# richness, px, py, pz, in the same order of halo catalog


# len(r[r < rvir[ih]*1e-3])


# sel=  (abs(x_gal - x_halo[ih]) < 2)&(abs(y_gal - y_halo[ih]) < 2 ) & (abs(z_gal - z_halo[ih]) < 2)
# len(r[sel])

# plt

# r = (x_gal - x_halo[ih])**2 + (y_gal - y_halo[ih])**2
# r = np.sqrt(r)
# d = abs(z_gal - z_halo[ih])
# sel = (r < rlam)&(d < 60)
# print(len(x_gal[sel]))

# nh = len(x_gal[sel])

#     #outfile.write('%i %e %i \n'%())
# outfile.close()