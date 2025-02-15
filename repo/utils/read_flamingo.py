#!/usr/bin/env python
import numpy as np
import h5py, fitsio
import os, sys
sys.path.append('../utils')
import yaml

class ReadFlamingo(object):
    def __init__(self, nbody_loc, redshift):
        self.input_loc = nbody_loc

        # get the cosmological parameters from header
        yml_fname = self.input_loc + 'used_parameters.yml'
        with open(yml_fname, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        cosmo = parsed_yaml['Cosmology']
        self.hubble = cosmo['h'] 
        self.OmegaM = 1. - cosmo['Omega_lambda']
        self.boxsize = 1000. * self.hubble # Todo: how to read boxsize from the header?
        #self.mpart = # defined later 
        self.redshift = redshift

        # get the snapshot name
        z_output = np.loadtxt(self.input_loc + 'output_list.txt')
        snap_id_list = np.arange(len(z_output))
        idx = np.argmin(abs(z_output-redshift))
        snap_id = snap_id_list[idx]
        self.snap_name = f'{snap_id:0>4d}'

        # calculate mpart ignoring gas. assuming all have the same mass
        #self.boxsize = max(self.xp) # Mpc/h
        fname = self.input_loc + f'snapshots_downsampled/flamingo_{self.snap_name}.hdf5'
        f = h5py.File(fname,'r')
        coord = f['DMParticles/Coordinates']
        npart = np.shape(coord)[0]
        
        rhocrit = 2.77536627e11 # h^2 Msun Mpc^-3
        total_mass_in_box_hiMsun = self.boxsize**3 * self.OmegaM * rhocrit
        self.mpart = total_mass_in_box_hiMsun / npart # Msun/h
        #print(f'npart={npart}, mpart={self.mpart:e}')

    def read_halos(self, Mmin, pec_vel=False, cluster_only=False, halo_loc='/cosma8/data/do012/dc-wu5/cylinder/output_HYDRO_PLANCK/'):
        # sigh... temporary solution for halo loc
        
        #if cluster_only == True:
        #    fname = self.input_loc+f'host_halos_{self.snap_name}_M12.5.fit'
        #else:
        fname = halo_loc + f'host_halos_{self.snap_name}.fit'
        data = fitsio.read(fname)
        Mvir = data['Mvir']
        sel = (Mvir >= Mmin)
        Mvir = Mvir[sel]
        sort = np.argsort(-Mvir)
        self.mass = Mvir[sort]

        self.hid = data['hid_host'][sel][sort]
        self.xh = data['px'][sel][sort]
        self.yh = data['py'][sel][sort]
        self.zh = data['pz'][sel][sort]

        if pec_vel == True: ## takes too long to read in velocity
            self.vx = data['vx'][sel][sort]
            self.vy = data['vy'][sel][sort]
            self.vz = data['vz'][sel][sort]


    def read_halos_orig(self, Mmin=1e11, pec_vel=False, cluster_only=False):
        
        fname = self.input_loc + f'SOAP/halo_properties_{self.snap_name}.hdf5'
        f = h5py.File(fname,'r')
        Mvir = f['SO/BN98/TotalMass'][:] * self.hubble # make it Msun/h
        sel = (Mvir >= Mmin)
        Mvir = Mvir[sel]
        sort = np.argsort(-Mvir)

        self.mass = Mvir[sort]

        self.hid = f['VR/ID'][sel][sort]

        pos = f['VR/CentreOfPotential']
        self.xh = pos[:,0][sel][sort] * self.hubble # make it Mpc/h
        self.yh = pos[:,1][sel][sort] * self.hubble
        self.zh = pos[:,2][sel][sort] * self.hubble

        if pec_vel == True:
            vel = f['BoundSubhaloProperties/CentreOfMassVelocity']
            self.vx = vel[:,0][sel][sort]
            self.vy = vel[:,1][sel][sort]
            self.vz = vel[:,2][sel][sort]

    def read_particle_positions(self):
        fname = self.input_loc + f'snapshots_downsampled/flamingo_{self.snap_name}.hdf5'
        f = h5py.File(fname,'r')
        coord = f['DMParticles/Coordinates']
        self.xp = coord[:,0] * self.hubble # make it Mpc/h
        self.yp = coord[:,1] * self.hubble
        self.zp = coord[:,2] * self.hubble



        return self.xp, self.yp, self.zp

    def read_particle_velocities(self):
        fname = self.input_loc + f'snapshots_downsampled/flamingo_{self.snap_name}.hdf5'
        f = h5py.File(fname,'r')
        vel = f['DMParticles/Velocities']
        self.vxp = vel[:,0]
        self.vyp = vel[:,1]
        self.vzp = vel[:,2]
        return self.vxp, self.vyp, self.vzp


if __name__ == '__main__':
    nbody_loc = f'/cosma8/data/dp004/flamingo/Runs/L1000N1800/HYDRO_PLANCK/'
    rfl = ReadFlamingo(nbody_loc=nbody_loc, redshift=0.3)
    # rfl.read_halos(Mmin=1e14, pec_vel=True)
    # print(len(rfl.xh))
    # print('max', np.max(rfl.vx))
    
    # rfl.read_particle_positions()
    # print(len(rfl.xp))
    # print('downsampled particle mass %e Msun/h'%rfl.mpart)

    # rfl.read_particle_velocities()
    # print(len(rfl.vxp))
