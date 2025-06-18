#!/usr/bin/env python
import os
import sys
import yaml

yml_fname = '../yml/flamingo/flamingo_galaxies.yml'
'''
##os.system(f'./calc_gal_den.py {yml_fname}')

os.system(f'./calc_richness_halo.py {yml_fname}')

##os.system(f'./plot_abundance.py {yml_fname}')
'''
binning = 'abundance_matching' # 'Ncyl'

os.system(f'./plot_lensing.py {yml_fname} {binning}')

