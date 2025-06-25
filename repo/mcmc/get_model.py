#!/usr/bin/env python
import numpy as np
import sys

#### Define the model class
sys.path.append('../emulator')
from s3_pred_radius import PredDataVector

class GetModel(object): 
    def __init__(self, emu_name, binning, iz, survey, params_free_name, params_fixed_value,
                 params_fixed_name, data_vector=['counts','lensing'], survey_area=1437, **kwargs):
        self.binning = binning
        
        self.data_vector = data_vector
        self.survey_area = survey_area
        self.pdv = PredDataVector(emu_name, binning, iz, survey, data_vector)
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
        
        # if self.binning == 'abun':
        #     model = self.pdv.pred_lensing(X_input)
        # else:
        #     model = np.append(self.pdv.pred_abundance(X_input), self.pdv.pred_lensing(X_input))

        model = []
        area_factor = self.survey_area / 1437.
        if 'counts' in self.data_vector:
            model.extend(self.pdv.pred_abundance(X_input) * area_factor)
        if 'lensing' in self.data_vector:
            model.extend(self.pdv.pred_lensing(X_input))

        return model


if __name__ == "__main__":
    emu_name = 'all'
    binning = 'abun'#'lam'
    survey = 'desy1thre'
    iz = 0
    zid = 3
    yml_name = f'yml/emcee_template.yml'
    from parse_yml import ParseYml
    parse = ParseYml(yml_name)
    nsteps, nwalkers, lsteps, burnin, params_free_name, params_free_ini, params_range,\
        params_fixed_name, params_fixed_value = parse.parse_yml()

    gm = GetModel(emu_name, binning, iz, survey, params_free_name, params_fixed_value, params_fixed_name, data_vector=['lensing'])
    

