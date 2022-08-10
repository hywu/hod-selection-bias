#!/usr/bin/env python
import os, sys

yml_fname = sys.argv[1]
depth = int(sys.argv[2])
#./pipeline.py yml/mini_uchuu_grid.yml 30


sys.path.append('../utils')
from read_yml import ReadYML
para = ReadYML(yml_fname)

def run_pipeline(model_id):
    print('doing model ', model_id)

    out_path = f'{para.output_loc}/model_{para.model_set}_{model_id}/'

    #### make galaxy catalog ####
    if os.path.exists(out_path+'gals.fit'):
        print('galaxies done')
    else:
        print('need galaxies')
        os.system(f'./make_gal_cat.py {yml_fname} {model_id}')

    #### calculate richness ####
    if os.path.exists(out_path+f'richness_d{depth}.fit'):
        print('richness done')
    else:
        print('need richness')
        os.system(f'./calc_richness.py {yml_fname} {model_id} {depth}')

    #### calculate lensing ####
    if os.path.exists(out_path+f'obs_d{depth}/DS_abun_bin_3.dat'):
        print('lensing done')
    else:
        from plot_lensing import PlotLensing
        plmu = PlotLensing(yml_fname, model_id, depth, abundance_matching=True, thresholded=False)
        plmu.calc_lensing()

    #### calculate counts vs. richness ####
    if os.path.exists(out_path+f'obs_d{depth}/counts_richness.dat'):
        print('counts done')
    else:
        from plot_counts_richness import PlotCountsRichness
        ccr = PlotCountsRichness(yml_fname, model_id, depth)
        ccr.calc_counts_richness()


for model_id in range(729):
    run_pipeline(model_id)