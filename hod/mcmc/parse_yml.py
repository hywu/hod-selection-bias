#!/usr/bin/env python

import numpy as np
import yaml

class ParseYml(object):
    def __init__(self, filename):
        self.filename = filename

    def parse_yml(self):

        '''
        Reading the mcmc setup from a yaml file
        '''
        self.params_free_ini = np.array([])
        lsteps = np.array([])
        
        self.params_free_range = np.array([])
        self.params_free_range = np.array([0, 0])

        self.params_free_name = []
        self.params_fixed_name = []

        self.params_fixed_value = np.array([])

        # for plotting purpuses
        self.params_free_label = [] # nice latex string
        self.params_free_truth = []

        with open(self.filename) as file:
            documents = yaml.load(file, Loader=yaml.SafeLoader)

        for i, item in enumerate(documents['params']):
            if item['vary'] is False:
                self.params_fixed_name.append(item['name'])
                self.params_fixed_value = np.append(self.params_fixed_value, item['value'])
                #continue
            else:
                self.params_free_name.append(item['name'])
                self.params_free_label.append(r"{}".format(item['label'])) # convert to a raw string
                self.params_free_truth = np.append(self.params_free_truth, item['truth']) 
            
                self.params_free_range = np.vstack([self.params_free_range, np.asarray(item['prior']['values'])])
                self.params_free_ini = np.append(self.params_free_ini, item['value'])
                lsteps = np.append(lsteps, item['lsteps'])

        self.params_free_range = self.params_free_range[1:]

        print('Free parameters:' + str(self.params_free_name))
        nsteps = documents['mcmc']['n_steps']
        nwalkers = documents['mcmc']['n_walkers']
        burnin = documents['mcmc']['burnin']
        return nsteps, nwalkers, lsteps, burnin, \
            self.params_free_name, self.params_free_ini, self.params_free_range,\
            self.params_fixed_name, self.params_fixed_value
        

if __name__ == "__main__":
    filename = 'yml/emcee_template.yml'
    parse = ParseYml(filename)
    parse.parse_yml()

    print('data_vector', parse.data_vector)

    print('params_free_name', parse.params_free_name)
    print('params_free_range', parse.params_free_range)
    print('params_free_ini', parse.params_free_ini)
    print('params_free_truth', parse.params_free_truth)
