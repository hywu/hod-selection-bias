#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
import os
import joblib

loc = '/projects/hywu/cluster_sims/cluster_finding/data/'
#emu_name = 'fixhod'
#emu_name = 'fixcos'

class PredDataVector(object):
    def __init__(self, emu_name, binning, iz, survey, data_vector=['lensing']): #'counts',
        #, survey='desy1thre'
        zid = 3+iz
        train_loc = loc + f'emulator_train/{emu_name}/z0p{zid}00/{survey}_{binning}/'
        self.train_loc = train_loc

        if survey == 'desy1':
            self.nbins = 4
        if survey == 'desy1thre':
            self.nbins = 1

        if 'counts' in data_vector:
            # emulator for abundance 
            self.gpr_abun_list = []
            for ilam in range(self.nbins):
                gpr = joblib.load(f'{train_loc}/abundance_bin_{ilam}_gpr_.pkl')
                self.gpr_abun_list.append(gpr)

        if 'lensing' in data_vector:
            # emulator for lensing
            # load all radii at the same time
            rp_rad = np.loadtxt(f'{train_loc}/rp_rad.dat')
            self.nrad = len(rp_rad)
            self.gpr_lensing_list = []
            for ilam in range(self.nbins):
                for irad in range(self.nrad):
                    gpr = joblib.load(f'{train_loc}/DS_{binning}_bin_{ilam}_rad_{irad}_gpr_.pkl')
                    self.gpr_lensing_list.append(gpr)


    def pred_abundance(self, X_input):
        X_input = np.atleast_2d(X_input)
        pred_list = []
        for ilam in range(self.nbins):
            pred_list.extend(self.gpr_abun_list[ilam].predict(X_input))
        return np.exp(pred_list)

    def pred_lensing(self, X_input):
        X_input = np.atleast_2d(X_input)
        pred_list = []
        for ilam in range(self.nbins):
            for irad in range(self.nrad):
                i = ilam * self.nrad + irad
                pred_list.extend(self.gpr_lensing_list[i].predict(X_input))
        return np.exp(pred_list)

if __name__ == "__main__":
    # load some cosmo parameters
    #emu_name='fixcos'
    emu_name = 'all'
    binning = 'abun'
    iz = 0
    zid = 3
    pdv = PredDataVector(emu_name, binning, iz)
    train_loc = pdv.train_loc
    #train_loc = loc + f'emulator_train/{emu_name}/z0p{zid}00/{binning}/'

    data = np.loadtxt(f'{train_loc}/parameters_all.dat')
    X_all = data[:,1:]
    itest = 0
    X_test = np.atleast_2d(X_all[itest]) # just one set of parameters

    ####
    
    #### lensing
    DS_pred = pdv.pred_lensing(X_test)
    DS_test = []
    for ilam in range(pdv.nbins):
        data = np.loadtxt(f'{train_loc}/DS_{binning}_bin_{ilam}_rad.dat')
        DS_test.extend(np.exp(data[itest]))
    #print('len(DS_test)', len(DS_test))
    print('DS err', (DS_pred - DS_test)/DS_test)
    '''
    #### abundance
    abun_pred = pdv.pred_abundance(X_test)
    data = np.loadtxt(f'{train_loc}/abundance.dat')
    abun_test = np.exp(data[itest])
    #print(abun_test)
    #print(abun_pred)
    print('abun err', (abun_pred - abun_test)/abun_test)
    '''