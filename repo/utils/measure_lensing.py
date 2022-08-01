import os
import numpy as np
from scipy.interpolate import interp1d

from Corrfunc._countpairs import countpairs_rp_pi


rpbins_loc = './'  # well this seems to work best
Rmin = 0.01
Rmax = 100
pimax = 100
n_decade = (np.log10(Rmax)-np.log10(Rmin))
nrp_per_decade = 5
n_rp = int(nrp_per_decade*n_decade + 1 + 0.001) ## adding the inner most bin


def write_bin_file():
    lnrp = np.linspace(np.log(Rmin), np.log(Rmax), n_rp)
    rp = np.exp(lnrp)
    outfile = open(rpbins_loc+'rp_bins.dat','w')
    outfile.write('%g %g \n'%(1e-6, rp[0]))
    for ir in range(n_rp-1):
        outfile.write('%g %g \n'%(rp[ir], rp[ir+1]))
    outfile.close()

if os.path.exists(rpbins_loc+'rp_bins.dat')==False:
    print('write rp_bins')
    write_bin_file()

def measure_lensing(xh, yh, zh, xp, yp, zp, boxsize, mpart):#boxsize=1100, mpart=None):
    nthreads = 4
    binfile = rpbins_loc+'rp_bins.dat'
    

    results_DD = countpairs_rp_pi(autocorr=0, nthreads=nthreads, pimax=pimax, binfile=binfile, X1=np.float64(xh), Y1=np.float64(yh), Z1=np.float64(zh), X2=np.float64(xp), Y2=np.float64(yp), Z2=np.float64(zp), boxsize=np.float64(boxsize), output_rpavg=True, periodic=True, verbose=False)[0]
    results_DD = np.array(results_DD)
    # rmin, rmax, rpavg, pi_upper, npairs, weight_avg

    #rp_min_2d = results_DD[:,0].reshape([n_rp, n_pi])
    rmin_flatten = results_DD[:,0]
    rmax_flatten = results_DD[:,1]
    rpsum_flatten = results_DD[:,2] * results_DD[:,4]
    pi_upper_flatten = results_DD[:,3]
    npairs_flatten = results_DD[:,4]#.reshape([n_rp, n_pi])

    select = (pi_upper_flatten <= pimax)
    npairs_2d = npairs_flatten[select].reshape([n_rp, pimax])
    rpsum_2d = rpsum_flatten[select].reshape([n_rp, pimax])

    npairs = np.sum(npairs_2d, axis=1)
    rpavg = np.sum(rpsum_2d, axis=1)/(1.*npairs)


    rpmin_list, rpmax_list = np.loadtxt(rpbins_loc+'rp_bins.dat', unpack=True)
    # if mpart == None: # Abacus's mass
    #     rhocrit = 2.775e11
    #     OmegaM = 0.314
    #     mpart = rhocrit * OmegaM * boxsize**3 / 1440**3
    #     mpart =  1000. * mpart

    # interplate Sigma(at rp) vs. rp_avg
    nhalo = len(xh)
    npart = npairs
    Sigma_at_rpavg = mpart * npart * 1./np.pi/(rpmax_list**2 - rpmin_list**2)/(nhalo*1.)
    lnSigma_at_lnrp_interp = interp1d(np.log(rpavg), np.log(Sigma_at_rpavg))

    # interplate Sigma(<rpmax) vs. rpmax
    Sigma_lt_rpmax = np.zeros(n_rp)
    for ir in range(n_rp):
        Sigma_lt_rpmax[ir] = mpart * sum(npart[0:ir+1])*1./np.pi/(rpmax_list[ir]**2)/(nhalo*1.)
    lnSigma_lt_lnrp_interp = interp1d(np.log(rpmax_list), np.log(Sigma_lt_rpmax))

    # get Sigma vs. rpmid, discard first bin
    rpmid_list = np.sqrt(rpmin_list*rpmax_list)[5:-1] # inner bins have NaN's
    # print(np.log(rpavg))
    # print(np.log(Sigma_at_rpavg))
    # print(np.log(rpmid_list[0]))
    Sigma_at_rp = np.exp(lnSigma_at_lnrp_interp(np.log(rpmid_list)))
    Sigma_lt_rp = np.exp(lnSigma_lt_lnrp_interp(np.log(rpmid_list)))

    DeltaSigma = (Sigma_lt_rp - Sigma_at_rp)
    
    return rpmid_list, Sigma_at_rp, DeltaSigma
# def measure_Sigma(xh, yh, zh, xp, yp, zp):
#     nthreads = 4
#     binfile = rpbins_loc+'rp_bins.dat'
#     boxsize = 1100

#     results_DD = countpairs_rp_pi(autocorr=0, nthreads=nthreads, pimax=pimax, binfile=binfile, X1=np.float64(xh), Y1=np.float64(yh), Z1=np.float64(zh), X2=np.float64(xp), Y2=np.float64(yp), Z2=np.float64(zp), boxsize=np.float64(boxsize), output_rpavg=True, periodic=True, verbose=False)[0]
#     results_DD = np.array(results_DD)
#     # rmin, rmax, rpavg, pi_upper, npairs, weight_avg

#     #rp_min_2d = results_DD[:,0].reshape([n_rp, n_pi])
#     rmin_flatten = results_DD[:,0]
#     rmax_flatten = results_DD[:,1]
#     rpsum_flatten = results_DD[:,2] * results_DD[:,4]
#     pi_upper_flatten = results_DD[:,3]
#     npairs_flatten = results_DD[:,4]#.reshape([n_rp, n_pi])

#     select = (pi_upper_flatten <= pimax)
#     npairs_2d = npairs_flatten[select].reshape([n_rp, pimax])
#     rpsum_2d = rpsum_flatten[select].reshape([n_rp, pimax])

#     npairs = np.sum(npairs_2d, axis=1)
#     rpavg = np.sum(rpsum_2d, axis=1)/(1.*npairs)


#     rp_min, rp_max = np.loadtxt(rpbins_loc+'rp_bins.dat', unpack=True)
#     area_annulus = np.pi * (rp_max**2 - rp_min**2)
#     rhocrit = 2.775e11
#     OmegaM = 0.314
#     mpart = rhocrit * OmegaM * boxsize**3 / 1440**3
#     #print('mpart = %e'%mpart)
#     mpart =  1000. * mpart / len(xh)
#     den = npairs / area_annulus * mpart
#     #plt.loglog(rpavg, den)

#     return rpavg, den