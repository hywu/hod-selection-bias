#!/usr/bin/env python
import yaml
import numpy as np
from numpy.random import default_rng
from scipy import spatial
from periodic_boundary_condition import periodic_boundary_condition

class DrawSatPositionsFromParticlesLayer(object):
    def __init__(self, yml_fname):#, pz_min, pz_max):
        with open(yml_fname, 'r') as stream:
            try:
                para = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.redshift = para['redshift']
        self.mdef = para['mdef']
        self.Om0 = para['OmegaM']

        seed = para.get('seed', 42)
        self.rng = default_rng(seed)

        ## get all particles
        if para['nbody'] == 'mini_uchuu':
            from read_mini_uchuu import ReadMiniUchuu
            self.readcat = ReadMiniUchuu(para['nbody_loc'], self.redshift)

        if para['nbody'] == 'uchuu':
            from read_uchuu import ReadUchuu
            self.readcat = ReadUchuu(para['nbody_loc'], self.redshift)

        boxsize = self.readcat.boxsize
        # x_padding = 2
        # y_padding = 2
        # z_padding = 2
        # xp_in, yp_in, zp_in = readcat.read_particles(pec_vel=True)
        # self.xp_all, self.yp_all, self.zp_all, self.vxp_all, self.vyp_all, self.vzp_all = periodic_boundary_condition(xp_in, yp_in, zp_in, boxsize, x_padding, y_padding, z_padding, pec_vel=True, vx=readcat.vxp, vy=readcat.vyp, vz=readcat.vzp)
        # print('finish reading particles')


    def particle_in_one_layer(self, pz_min, pz_max):
        self.xp_layer, self.yp_layer, self.zp_layer, self.vxp_layer, self.vyp_layer, self.vzp_layer = self.readcat.read_particles_layer(pz_min, pz_max)
        # z_padding = 2
        # sel_part = (self.zp > pz_min - z_padding)&(self.zp < pz_max + z_padding)

        # self.xp_layer = self.xp_all[sel_part]
        # self.yp_layer = self.yp_all[sel_part]
        # self.zp_layer = self.zp_all[sel_part]
        # self.vxp_layer = self.vxp_all[sel_part]
        # self.vyp_layer = self.vyp_all[sel_part]
        # self.vzp_layer = self.vzp_all[sel_part]

        part_position = np.dstack([self.xp_layer, self.yp_layer, self.zp_layer])[0]
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
        Npart = len(self.xp_layer[indx])

        if Npart < Nsat: # argh...find too few particles due to subsampling...
            Ndraw = Npart
            Nextra = Nsat - Ndraw
        else:
            Ndraw = Nsat
            Nextra = 0

        sel = np.random.randint(0, Npart, Ndraw)
        x_sat = self.xp_layer[indx][sel]
        y_sat = self.yp_layer[indx][sel]
        z_sat = self.zp_layer[indx][sel]
        vx_sat = self.vxp_layer[indx][sel]
        vy_sat = self.vyp_layer[indx][sel]
        vz_sat = self.vzp_layer[indx][sel]

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
    yml_fname = '../yml/mini_uchuu/mini_uchuu_fid_hod.yml'
    # x_cen = 255.24039
    # y_cen = 117.69403
    # z_cen = 246.40901
    x_cen = 34.38733 # this halo is at the 0-4 layer
    y_cen = 250.40297 
    z_cen = 3.92614
    dsp = DrawSatPositionsFromParticlesLayer(yml_fname)
    dsp.particle_in_one_layer(0.0, 4.0)
    x, y, z, vx, vy, vz = dsp.draw_sats(1e14, 100, x_cen, y_cen, z_cen)
    print('x', x)
    print('y', y)
    print('z', z)