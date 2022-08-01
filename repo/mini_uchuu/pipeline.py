#!/usr/bin/env python
import os


def run_pipeline(model_id):
    print('doing model ', model_id)

    depth = 30
    output_loc = '/bsuhome/hwu/scratch/hod-selection-bias/output_mini_uchuu/'
    out_path = f'{output_loc}/model_{model_id}/'

    #### make galaxy catalog ####
    if os.path.exists(out_path+'gals.fit'):
        print('galaxies done')
    else:
        print('need galaxies')
        os.system(f'./make_gal_cat.py --which_sim mini_uchuu --model_id {model_id}')

    #### calculate richness ####
    if os.path.exists(out_path+f'richness_d{depth}.fit'):
        print('richness done')
    else:
        print('need richness')
        os.system(f'./calc_richness.py --which_sim mini_uchuu --model_id {model_id} --depth {depth}')

    #### calculate lensing ####
    if os.path.exists(out_path+f'obs_d{depth}/DS_abun_bin_3.dat'):
        print('lensing done')
    else:
        from plot_lensing import PlotLensing
        plmu = PlotLensing('mini_uchuu', model_id, depth, abundance_matching=True, thresholded=False)
        plmu.calc_lensing()

    # #### calculate counts vs. richness ####
    # if os.path.exists(out_path+f'obs_d{depth}/counts_richness.dat'):
    #     print('counts done')
    # else:
    #     from calc_counts_richness import CalcCountsRichness
    #     ccr = CalcCountsRichness('mini_uchuu', model_id, depth)
    #     ccr.calc_counts_richness()


for model_id in [4]:#range():
    run_pipeline(model_id)