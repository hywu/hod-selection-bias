#!/usr/bin/env python
import fitsio
from astropy.io import fits
from astropy.table import join
from astropy.table import Table
import numpy as np
import yaml
import sys

## first run ./calc_richness_rank.py ../yml/flamingo/flamingo_env.yml
## with no ranking and no percolation


yml_fname = sys.argv[1]

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

depth = para['depth']
output_loc = para['output_loc']
model_name = para['model_name']
#rich_name = para['rich_name']

out_path = f'{output_loc}/model_{model_name}/'
env_proxy = para['rank_proxy']

def create_env_rank_file():
    #### original gal catalog ####
    data_gal, header = fitsio.read(f'{out_path}/gals.fit', header=True)
    id1 = data_gal['hid_sub']
    #print(len(id1))
    x = data_gal['px']
    y = data_gal['py']
    z = data_gal['pz']

    #### environmental proxies
    data, header = fitsio.read(f'{out_path}/richness_{env_proxy}_env.fit', header=True)
    id2 = data['cluster_id']
    env = data['lambda']
    #print(len(id2))

    #### join by ID ####
    from astropy.table import QTable
    data1 = QTable([id1, x, y, z], names=('id', 'x', 'y', 'z'))
    data2 = QTable([id2, env], names=('id', 'env'))

    
    data_joined = join(data1, data2, keys=['id'])
    
    #print(data_joined.keys())
    
    id3 = data_joined['id']
    env = data_joined['env']
    x = data_joined['x']
    y = data_joined['y']
    z = data_joined['z']
    
    #print(len(id1), len(id2), len(id3))

    print(max(abs(id1-id3))) # merged file have the order as data1
    print(max(abs(id2-id3)))


    # Append env to the galaxy fits file
    with fits.open(f'{out_path}/gals.fit') as hdul:
        data = hdul[1].data
        table = Table(data)
        table['env'] = env 
        table['px_test'] = x 
        table['id_test'] = id3 
        new_hdu = fits.BinTableHDU(table)
        new_hdul = fits.HDUList(hdul[0:1] + [new_hdu] + hdul[2:])
        new_hdul.writeto(f'{out_path}/gals_{env_proxy}_env.fit', overwrite=True)

        #### sanity check using px_test
        data_test = fitsio.read(f'{out_path}/gals_{env_proxy}_env.fit')
        x1 = data_test['px']
        x2 = data_test['px_test']
        id1 = data_test['hid_sub']
        id2 = data_test['id_test']
        print(max(abs(x1-x2))) # should be zero
        print(max(abs(id1-id2))) # should be zero

create_env_rank_file()