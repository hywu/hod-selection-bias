#!/usr/bin/env python
import os
import sys
import yaml

#./pipeline.py yml/mini_uchuu_fid_hod.yml
#./pipeline.py yml/abacus_summit_fid_hod.yml

yml_fname = sys.argv[1]

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

depth = para['depth']
output_loc = para['output_loc']
model_name = para['model_name']
out_path = f'{output_loc}/model_{model_name}/'

#### make galaxy catalog ####
if os.path.exists(out_path+'gals.fit'):
    print('galaxies done')
else:
    print('need galaxies')
    os.system(f'./make_gal_cat.py {yml_fname}')

#### calculate richness ####
if os.path.exists(out_path+f'richness_d{depth}.fit'):
    print('richness done')
else:
    print('need richness')
    os.system(f'./calc_richness.py {yml_fname}')

#### calculate lensing ####
if os.path.exists(out_path+f'obs_d{depth}/DS_abun_bin_3.dat'):
    print('lensing done')
else:
    from plot_lensing import PlotLensing
    plmu = PlotLensing(yml_fname, abundance_matching=True, thresholded=False)
    plmu.calc_lensing()

#### calculate counts vs. richness ####
if os.path.exists(out_path+f'obs_d{depth}/counts_richness.dat'):
    print('counts done')
else:
    from plot_counts_richness import PlotCountsRichness
    ccr = PlotCountsRichness(yml_fname)#, model_id, depth)
    ccr.calc_counts_richness()

