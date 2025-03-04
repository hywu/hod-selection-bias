#!/usr/bin/env python
import timeit
start = timeit.default_timer()
start_master = start * 1
import os
import sys
import yaml

#./pipeline.py ../yml/mini_uchuu/mini_uchuu_fid_hod.yml

yml_fname = sys.argv[1]

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#### For AbacusSummit ####
if para['nbody'] == 'abacus_summit':
    cosmo_id = para.get('cosmo_id', None)
    hod_id = para.get('hod_id', None)
    phase = para.get('phase', None)
    redshift = para['redshift']
    if redshift == 0.3: z_str = '0p300'
    output_loc = para['output_loc']+f'/base_c{cosmo_id:0>3d}_ph{phase:0>3d}/z{z_str}/'
else:
   output_loc = para['output_loc']
   redshift = para['redshift']



depth = para['depth']
model_name = para['model_name']
rich_name = para['rich_name']
out_path = f'{output_loc}/model_{model_name}/'
# los_in = para.get('los', 'z')
# if los_in == 'xyz':
#     los_list = ['x', 'y', 'z']
# else:
#     los_list = ['z']

print('rich_name', rich_name)

#### make galaxy catalog ####
if os.path.exists(out_path+'gals.fit'):
    print('galaxies done')
else:
    print('need galaxies')
    os.system(f'./make_gal_cat.py {yml_fname}')



#### calculate richness ####
#for los in los_list:
#if los == 'z':
fname = out_path+f'richness_{rich_name}.fit'
#else:
#    fname = out_path+f'richness_{rich_name}.fit'
if os.path.exists(fname):
    print('richness done')
else:
    print('need richness')
    os.system(f'./calc_richness_halo.py {yml_fname}')

#### calculate lensing ####

#if los == 'z':
#obs_path = out_path+f'obs_{rich_name}/'
survey = para.get('survey', 'desy1')
obs_path = f'{out_path}/obs_{rich_name}_{survey}/'

#else:
#    obs_path = out_path+f'obs_d{depth}_{los}/'
if survey == 'desy1':
    lens_fname = obs_path+'DS_abun_bin_3.dat'
if survey == 'sdss':
    lens_fname = obs_path+'DS_abun_bin_0.dat'

if os.path.exists(lens_fname):
    print('lensing done')
else:
    print('need lensing')
    os.system(f'./plot_lensing.py {yml_fname}')
    #from plot_lensing import PlotLensing
    #plmu = PlotLensing(yml_fname, abundance_matching=True, thresholded=False)
    #plmu.calc_lensing()

#### calculate counts vs. richness ####
if os.path.exists(obs_path+'/counts_richness.dat'):
    print('counts done')
else:
    os.system(f'./plot_counts_richness.py {yml_fname}')
    # sys.path.append('../plots_for_paper/')
    # from plot_counts_richness import PlotCountsRichness
    # ccr = PlotCountsRichness(yml_fname)#, model_id, depth)
    # ccr.calc_counts_richness()


# if los_in == 'xyz':
#     os.system(f'./avg_xyz_lensing.py {yml_fname}')


stop = timeit.default_timer()
dtime = (stop - start_master)/60.
print(f'total time {dtime:.2g} mins')
