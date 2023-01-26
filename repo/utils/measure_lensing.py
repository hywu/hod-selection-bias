#!/usr/bin/env python
import os
import numpy as np
from scipy.interpolate import interp1d
from Corrfunc._countpairs import countpairs_rp_pi

class MeasureLensing(object):
    def __init__(self, out_loc, Rmin, Rmax, pimax): # pimax cannot be too large for TNG 
        self.out_loc = out_loc
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.pimax = pimax

        n_decade = (np.log10(Rmax)-np.log10(Rmin))
        nrp_per_decade = 5
        self.n_rp = int(nrp_per_decade*n_decade + 1 + 0.001) ## adding the inner most bin
        self.binfile = self.out_loc+'rp_bins.dat'

    def write_bin_file(self):
        lnrp = np.linspace(np.log(self.Rmin), np.log(self.Rmax), self.n_rp)
        rp = np.exp(lnrp)
        outfile = open(self.binfile, 'w')
        outfile.write('%g %g \n'%(0, rp[0]))
        for ir in range(self.n_rp-1):
            outfile.write('%g %g \n'%(rp[ir], rp[ir+1]))
        outfile.close()

    def measure_lensing(self, xh, yh, zh, xp, yp, zp, boxsize, mpart):
        nthreads = 12

        results_DD = countpairs_rp_pi(autocorr=0, nthreads=nthreads, pimax=self.pimax, binfile=self.binfile, 
            X1=xh.astype(float), Y1=yh.astype(float), Z1=zh.astype(float), 
            X2=xp.astype(float), Y2=yp.astype(float), Z2=zp.astype(float), 
            boxsize=np.float64(boxsize), output_rpavg=True, 
            periodic=True, verbose=False)[0]

        # X1=np.float64(xh), Y1=np.float64(yh), Z1=np.float64(zh), 
        # X2=np.float64(xp), Y2=np.float64(yp), Z2=np.float64(zp), 


        results_DD = np.array(results_DD)
        # rmin, rmax, rpavg, pi_upper, npairs, weight_avg
        #rp_min_2d = results_DD[:,0].reshape([n_rp, n_pi])
        rmin_flatten = results_DD[:,0]
        rmax_flatten = results_DD[:,1]
        rpsum_flatten = results_DD[:,2] * results_DD[:,4]
        pi_upper_flatten = results_DD[:,3]
        npairs_flatten = results_DD[:,4]#.reshape([n_rp, n_pi])

        select = (pi_upper_flatten <= self.pimax)
        npairs_2d = npairs_flatten[select].reshape([self.n_rp, self.pimax])
        rpsum_2d = rpsum_flatten[select].reshape([self.n_rp, self.pimax])

        npairs = np.sum(npairs_2d, axis=1)
        rpavg = np.sum(rpsum_2d, axis=1)/(1.*npairs)

        rpmin_list, rpmax_list = np.loadtxt(self.binfile, unpack=True)
        # if mpart == None: # Abacus's mass
        #     rhocrit = 2.775e11
        #     OmegaM = 0.314
        #     mpart = rhocrit * OmegaM * boxsize**3 / 1440**3
        #     mpart =  1000. * mpart

        # interplate Sigma(at rp) vs. rp_avg
        nhalo = len(xh)
        #npart = npairs
        Sigma_at_rpavg = mpart * npairs * 1./np.pi/(rpmax_list**2 - rpmin_list**2)/(nhalo*1.)
        lnSigma_at_lnrp_interp = interp1d(np.log(rpavg), np.log(Sigma_at_rpavg))

        # interplate Sigma(<rpmax) vs. rpmax
        Sigma_lt_rpmax = np.zeros(self.n_rp)
        for ir in range(self.n_rp):
            Sigma_lt_rpmax[ir] = mpart * sum(npairs[0:ir+1])*1./np.pi/(rpmax_list[ir]**2)/(nhalo*1.)
        lnSigma_lt_lnrp_interp = interp1d(np.log(rpmax_list), np.log(Sigma_lt_rpmax))

        # get Sigma vs. rpmid, discard first few bins
        rpmid_list = np.sqrt(rpmin_list*rpmax_list)[5:-1] # inner bins have NaN's
        # print(np.log(rpavg))
        # print(np.log(Sigma_at_rpavg))
        # print(np.log(rpmid_list[0]))
        Sigma_at_rp = np.exp(lnSigma_at_lnrp_interp(np.log(rpmid_list)))
        Sigma_lt_rp = np.exp(lnSigma_lt_lnrp_interp(np.log(rpmid_list)))

        DeltaSigma = (Sigma_lt_rp - Sigma_at_rp)
        
        return rpmid_list, Sigma_at_rp, DeltaSigma
        #return rpavg, Sigma_at_rpavg, DeltaSigma

if __name__ == "__main__":
    #### demo code ####
    import sys
    sys.path.append('../utils')
    from read_mini_uchuu import ReadMiniUchuu
    ouput_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
    nbody_loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'
    rmu = ReadMiniUchuu(nbody_loc)
    xp, yp, zp = rmu.read_particles()

    #### halos ####
    rmu.read_halos(Mmin=1e13)
    boxsize = rmu.boxsize
    xh = rmu.xh
    yh = rmu.yh
    zh = rmu.zh
    mass = rmu.mass
    boxsize = rmu.boxsize
    rhocrit = 2.775e11
    OmegaM = rmu.OmegaM
    mpart = rhocrit * OmegaM * boxsize**3 / len(xp)

    Mmin = 1e14
    Mmax = 2e14
    sel = (mass > Mmin)&(mass < Mmax)
    out_loc = 'data/'
    ml = MeasureLensing(out_loc, Rmin=0.1, Rmax=1, pimax=10)
    ml.write_bin_file()
    rp, Sigma, DeltaSigma = ml.measure_lensing(
        xh[sel], yh[sel], zh[sel], xp, yp, zp, boxsize, mpart)

