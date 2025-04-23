#!/usr/bin/env python
import numpy as np
import sys

#### Define the model class
sys.path.append('../emulator')
from s3_pred_radius import PredDataVector

class GetModel(object): 
    def __init__(self, emu_name, iz, params_free_name, params_fixed_value,
                 params_fixed_name, **kwargs):
        self.pdv = PredDataVector(emu_name, iz)
        self.params_fixed_name = params_fixed_name
        self.params_free_name = params_free_name
        self.params_fixed_value = params_fixed_value
       
    def get_kw(self, params):
        kw = {} # if there's extra keyword
        for i, pf in enumerate(self.params_fixed_name):
            kw[pf] = self.params_fixed_value[i]

        for i, pf in enumerate(self.params_free_name):
            kw[pf] = params[i]
        ###
        self.kw = kw
    
    def model(self, params):
        self.get_kw(params)
        params = self.kw
        #print('params inside model', params)
        sigma8 = params['sigma8']
        OmegaM = params['OmegaM']
        ns = params['ns']
        OmegaB = params['OmegaB']
        w0 = params['w0']
        wa = params['wa']
        #sigma8, OmegaM, ns, Ob0, w0, wa, Nur, alpha_s
        cosmo_para = np.array([sigma8, OmegaM, ns, OmegaB, w0, wa, 2.0328, 0])
        
        alpha = params['alpha']
        lgM1 = params['lgM1']
        lgMcut = params['lgMcut']
        # alpha,lgM1,lgkappa,lgMcut,sigmalogM
        hod_para = np.array([alpha, lgM1, 0, lgMcut, 0.1]) #1,12.9,0,11.7,0.1


        X_input = np.append(cosmo_para, hod_para)
        model = np.append(self.pdv.pred_abundance(X_input), self.pdv.pred_lensing(X_input))
        return model

