#!/usr/bin/env python
import os


#sim_name0 = 'L1000N1800/HYDRO_PLANCK'
# sim_name1 = 'L1000N0900/HYDRO_FIDUCIAL'
# sim_name2 = 'L1000N1800/HYDRO_FIDUCIAL'
# sim_name3 = 'L1000N3600/HYDRO_FIDUCIAL'

# for sim_name in [sim_name1, sim_name2, sim_name3]:
#     os.system(f'./calc_HMF_SMF_LF.py {sim_name}')
#     #os.system(f'./make_gal_cat_mstar.py {sim_name}')
#     #os.system(f'./make_gal_cat_red.py {sim_name}')


# DMO HMF
sim_name1 = 'L1000N0900/DMO_FIDUCIAL'
sim_name2 = 'L1000N1800/DMO_FIDUCIAL'
sim_name3 = 'L1000N3600/DMO_FIDUCIAL'
from calc_HMF_SMF_LF import calc_HMF
for halofinder in ['VR', 'HBT']:
    for mdef in ['vir', '200m']:
        for sim_name in [sim_name1, sim_name2, sim_name3]:
            calc_HMF(sim_name, halofinder, mdef)