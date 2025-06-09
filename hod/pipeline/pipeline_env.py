#!/usr/bin/env python
import os
import sys
import yaml

yml_fname1 = '../yml/flamingo/flamingo_env_step1.yml'
yml_fname2 = '../yml/flamingo/flamingo_env_step2.yml'

#os.system(f'./calc_richness_rank.py {yml_fname1}')

#os.system(f'./create_env_rank.py {yml_fname2}')

os.system(f'./calc_richness_rank.py {yml_fname2}')

# os.system(f'./cluster_halo_matching.py {yml_fname2}')

# from plot_lensing import PlotLensing
# plmu = PlotLensing(yml_fname2, abundance_matching=True, thresholded=False)
# plmu.calc_lensing()
