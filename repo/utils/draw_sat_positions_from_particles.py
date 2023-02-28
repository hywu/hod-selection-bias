#!/usr/bin/env python
import yaml
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy import spatial
from periodic_boundary_condition import periodic_boundary_condition

# from scipy.interpolate import interp1d
# from colossus.cosmology import cosmology
# from colossus.halo import concentration


## TODO! need periodic boundary condition


class DrawSatPositionsFromParticles(object):
    def __init__(self, yml_fname):
        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.redshift = para['redshift']
        self.mdef = para['mdef']
        self.Om0 = para['OmegaM']
        # Ob0 = para['OmegaB']
        # sigma8 = para['sigma8']
        # ns = para['ns']
        # H0 = para['hubble'] * 100

        seed = para.get('seed', 42)
        self.rng = default_rng(seed)

        ## get all particles
        if para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            readcat = ReadMiniUchuu(para['nbody_loc'], self.redshift)

        xp_in, yp_in, zp_in = readcat.read_particles(pec_vel=True)
        x_padding = 2
        y_padding = 2
        z_padding = 2
        boxsize = readcat.boxsize

        self.xp, self.yp, self.zp, self.vxp, self.vyp, self.vzp = periodic_boundary_condition(xp_in, yp_in, zp_in, boxsize, x_padding, y_padding, z_padding, pec_vel=True, vx=readcat.vxp, vy=readcat.vyp, vz=readcat.vzp)

        part_position = np.dstack([self.xp, self.yp, self.zp])[0]
        self.part_tree = spatial.cKDTree(part_position)

    def draw_sats(self, mass, Nsat, x_cen, y_cen, z_cen):
        rhocrit = 2.775e11 # h-unit
        if self.mdef == '200m':
            radius = (3 * mass / (4 * np.pi * rhocrit * self.Om0 * 200.))**(1./3.) # cMpc/h (chimp)
        elif self.mdef == 'vir':
            OmegaM_z = self.Om0 * (1+ self.redshift)**3 / (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)
            x = OmegaM_z - 1
            Delta_vir_c = 18 * np.pi**2 + 82 * x - 39 * x**2
            rhocrit_z = rhocrit * (self.Om0 * (1+ self.redshift)**3 + 1 - self.Om0)/(1+self.redshift)**3 # gotcha!
            radius = (3 * mass / (4 * np.pi * rhocrit_z * Delta_vir_c))**(1./3.)
        else:
            print('need to implement radius')

        indx = self.part_tree.query_ball_point([x_cen, y_cen, z_cen], radius)
        Npart = len(self.xp[indx])

        if Npart < Nsat: # argh...find too few particles due to subsampling...
            Ndraw = Npart
            Nextra = Nsat - Ndraw
        else:
            Ndraw = Nsat
            Nextra = 0

        sel = np.random.randint(0, Npart, Ndraw)
        x_sat = self.xp[indx][sel]
        y_sat = self.yp[indx][sel]
        z_sat = self.zp[indx][sel]
        vx_sat = self.vxp[indx][sel]
        vy_sat = self.vyp[indx][sel]
        vz_sat = self.vzp[indx][sel]

        if Nextra > 0:
            print(f'problematic halo {mass:e} Npart={Npart} Nsat={Nsat}')
            x_extra = np.array([x_cen]) + np.zeros(Nextra)
            y_extra = np.array([y_cen]) + np.zeros(Nextra)
            z_extra = np.array([z_cen]) + np.zeros(Nextra)
            G = 4.302e-9 # Mpc/Msun (km/s)^2
            v_std = (2./3.) * G * mass / radius
            v_std = np.sqrt(v_std)
            vx_extra, vy_extra, vz_extra = np.random.normal(0, v_std, (3, Nextra))
            x_sat = np.append(x_sat, x_extra)
            y_sat = np.append(y_sat, y_extra)
            z_sat = np.append(z_sat, z_extra)
            vx_sat = np.append(vx_sat, vx_extra)
            vy_sat = np.append(vy_sat, vy_extra)
            vz_sat = np.append(vz_sat, vz_extra)

        return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat

if __name__ == "__main__":
    yml_fname = '../scripts/yml/mini_uchuu_fid_hod.yml'
    x_cen = 255.24039
    y_cen = 117.69403
    z_cen = 246.40901
    dsp = DrawSatPositionsFromParticles(yml_fname)
    x, *b = dsp.draw_sats(1e14, 100, x_cen, y_cen, z_cen)
    print(len(x))