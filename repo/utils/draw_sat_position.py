#!/usr/bin/env python
import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.interpolate import interp1d

from colossus.cosmology import cosmology
from colossus.halo import concentration

class DrawSatPosition(object):
    def __init__(self, yml_fname):
        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.redshift = para['redshift']
        self.mdef = para['mdef']
        self.Om0 = para['OmegaM']
        Ob0 = para['OmegaB']
        sigma8 = para['sigma8']
        ns = para['ns']
        H0 = para['hubble'] * 100

        seed = para.get('seed', 42)
        self.rng = default_rng(seed)

        params = {'flat': True, 'H0': H0, 'Om0': self.Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': ns}
        cosmo = cosmology.setCosmology('MyCosmo', params)
        mass_interp = 10**np.linspace(11, 15.5, 10)

        c_interp = concentration.concentration(mass_interp, z=self.redshift, mdef=self.mdef, model='bhattacharya13')
        self.c_lnM_interp = interp1d(np.log(mass_interp), c_interp)

    def draw_sat_position(self, mass, Nsat):
        rhocrit = 2.775e11 # h-unit
        if self.mdef == '200m':
            self.radius = (3 * mass / (4 * np.pi * rhocrit * self.Om0 * 200.))**(1./3.) # cMpc/h (chimp)
        elif self.mdef == 'vir':
            OmegaM_z = self.Om0 * (1+ self.redshift)**3 / (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)
            x = OmegaM_z - 1
            Delta_vir_c = 18 * np.pi**2 + 82 * x - 39 * x**2
            rhocrit_z = rhocrit * (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)/(1+self.redshift)**3 # gotcha!
            self.radius = (3 * mass / (4 * np.pi * rhocrit_z * Delta_vir_c))**(1./3.) 
            # # validated with colossus! Don't delete this.
            # from colossus.halo import mass_so
            # print('Delta_vir', Delta_vir_c, mass_so.deltaVir(self.redshift))
            # a = 1/(1+self.redshift)
            # R_colo = mass_so.M_to_R(mass, self.redshift, 'vir')/1e3 /a 
            # print('Rvir', radius, R_colo)
        else:
            print('need to implement radius')
        c = self.c_lnM_interp(np.log(mass))
        r_nfw = 10**np.linspace(-4, np.log10(self.radius), 100)  # chimp # lower lim can't be larger than -4.
        rs = self.radius / c
        cdf = 4 * np.pi * rs**3 * (np.log((r_nfw + rs)/rs) - r_nfw/(r_nfw + rs) ) / mass
        cdf /= cdf[-1]

        r_cdf_interp = interp1d(cdf, r_nfw)
        rand_vals = self.rng.random(Nsat)
        rsat_cMpc = r_cdf_interp(rand_vals)

        u = self.rng.random(Nsat)
        v = self.rng.random(Nsat)
        theta = 2 * np.pi * u 
        phi = np.arccos(2 * v - 1)
        px = rsat_cMpc * np.cos(theta) * np.sin(phi)
        py = rsat_cMpc * np.sin(theta) * np.sin(phi)
        pz = rsat_cMpc * np.cos(phi)

        #if max(rsat_cMpc) > radius: print('outside radius')
        #self.radius = radius
        return px, py, pz

    def draw_sat_velocity(self, mass, Nsat): # need to run draw_sat_position first to get radius
        G = 4.302e-9 # Mpc/Msun (km/s)^2
        v_std = (2./3.) * G * mass / self.radius
        v_std = np.sqrt(v_std)
        vx, vy, vz = np.random.normal(0, v_std, (3, Nsat))
        return vx, vy, vz

if __name__ == "__main__":
    yml_fname = '../scripts/yml/mini_uchuu_fid_hod.yml'
    dsp = DrawSatPosition(yml_fname)
    dsp.draw_sat_position(1e14, 100)
    vx, vy, vz = dsp.draw_sat_velocity(1e14, 100)
    print(np.std(vx))