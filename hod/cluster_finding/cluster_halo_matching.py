#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import fitsio
from astropy.table import Table
from astropy.io import fits
import sys
import yaml

# get the cluster id
# get the centers's hid_host, x_cen, y_cen
# get the members 
# find the best-matched hid_best, mass_best, x_best, y_best
# calculate the offset 
# category 1: cluster center = best-matched halo center (offset == 0)
# category 2: cluster center is a satellite of the best-matched halo (hid_host_cen == hid_best) 
# category 3: cluster center is NOT a satllite of the best-matched halo (hid_host_cen == hid_best)


yml_fname = sys.argv[1]

with open(yml_fname, 'r') as stream:
    try:
        para = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

depth = para['depth']
output_loc = para['output_loc']
model_name = para['model_name']

#output_loc = '/cosma8/data/do012/dc-wu5/cylinder/output_HYDRO_PLANCK/'
depth = para['depth']
output_loc = para['output_loc']
model_name = para['model_name']
#rich_name = para['rich_name']

out_path = f'{output_loc}/model_{model_name}/'
#env_proxy = para['rank_proxy'] #'d90_rp0.1'

rank_proxy =  para['rank_proxy'] #'rp0.1' #'lum' #'env' #

#### read in the clusters
richness_fname = f'{out_path}/richness_{rank_proxy}.fit'


data_cl, header_cl = fitsio.read(richness_fname, header=True)
#print(header_cl)

lam = data_cl['lambda']
rlam = data_cl['rlambda']

boxsize = 1000. * 0.6732

ncl = int(1.e-5 * boxsize**3)
lam_cut = -np.sort(-lam)[ncl]
print('lambda cut = ', lam_cut)

#lam_cut = 40 # for quick debugging
sel_rich = (lam >= lam_cut)
lam = lam[sel_rich]
rlam = rlam[sel_rich]
cluster_id = data_cl['cluster_id'][sel_rich]
x_cl = data_cl['px'][sel_rich]
y_cl = data_cl['py'][sel_rich]
hid_host_cen = data_cl['hid_host'][sel_rich]

ncl = len(cluster_id)
print('ncl', ncl)



# read in the members
data_mem, header_mem = fitsio.read(f'{output_loc}/model_redmagic/members_{rank_proxy}.fit', header=True)
#print(header)
cluster_id_mem = data_mem['cluster_id']

hid_best = []

#### find the best matched halo
for icl in range(ncl):
    # get the member of one cluster to work on
    cluster_id_wanted = cluster_id[icl]
    data_mem_1cl = data_mem[cluster_id_mem == cluster_id_wanted]

    haloid = data_mem_1cl['hid_host']
    pmem = data_mem_1cl['pmem']
    mass_host = data_mem_1cl['mass_host']

    haloid_unique = np.unique(haloid) # how many halos contribute to this cluster?
    nhalo = len(haloid_unique)
    #print('nhalo', nhalo)
    sum_pmem = np.zeros(nhalo)  # find the highest sum pmem
    for i in range(nhalo):
        sel = (haloid == haloid_unique[i])
        sum_pmem[i] = np.sum(pmem[sel])
        #print(haloid_unique[i], np.unique(mass_host[sel]), sum_pmem[i])
        
    #print(haloid_unique)
    #print(sum_pmem)
    arg = np.argmax(sum_pmem)
    #print('best-matched halo id', haloid_unique[arg])
    hid_best.append(haloid_unique[arg])


#### calculate the center offset
import sys
sys.path.append('/cosma/home/do012/dc-wu5/work/hod-selection-bias/repo/utils')
from periodic_boundary_condition import periodic_boundary_condition


## read in the best-matched halos and cluster halo
# find their Mvir, x, y from 'host_halos_xxx.fit'
### read in all halos 
data_h, header_h = fitsio.read(output_loc+f'/host_halos_0071.fit', header=True)
#print(header_h)
x_h_in = data_h['px']
y_h_in = data_h['py']
z_h_in = data_h['pz']
haloid_h_in = data_h['hid_host']
m_h_in = data_h['Mvir']

# apply the periodic boundary condition (some best matched halos are on the other side of the box...)
x_padding = 10 
y_padding = 10 
z_padding = 0
x_h, y_h, z_h, haloid_h, m_h = periodic_boundary_condition(x_h_in, y_h_in, z_h_in, boxsize, x_padding, y_padding, z_padding, haloid_h_in, m_h_in)

offset_list = []
mass_best = []

for icl in range(ncl): # [242]: at the boundary
    cluster_id_wanted = cluster_id[icl]
    hid_best_wanted = hid_best[icl]
    
    sel_h = (haloid_h == hid_best_wanted)

    if cluster_id_wanted == hid_best_wanted:
        offset = 0
    else:
        sel_c = (cluster_id == cluster_id_wanted)
        x_cl_wanted = x_cl[sel_c]
        y_cl_wanted = y_cl[sel_c]

        x_true = x_h[sel_h] # there may be 2 of 3 if the halo is near the boundary
        y_true = y_h[sel_h]
        #print(x_true, y_true)
        offset = (x_cl_wanted - x_true)**2 + (y_cl_wanted - y_true)**2 
        offset = np.sqrt(offset)
        #if len(offset)>1: print(offset)
        #print(icl, m_h[sel_h], offset)
        offset = min(offset) 
            
    offset_list.append(offset)
    mass_best.append(m_h[sel_h][0])

#exit()

#data = np.array([cluster_id, hid_host_cen, hid_best, offset_list, mass_best, lam, rlam]).transpose()
#np.savetxt(f'{output_loc}/model_redmagic/halo_best_d90_{rank_proxy}.dat', 
#    data, fmt='%-12i %-12i %-12i %-12g %-12e %-12g %-12g ', 
#    header='cluster_id hostid_cen hid_best offset mass_best lam rlam')

'''
from astropy.io import fits
cols=[
  fits.Column(name='cluster_id', format='K' ,array=cluster_id),
  fits.Column(name='hid_host_cen', format='K' ,array=hid_host_cen),
  fits.Column(name='hid_best', format='K' ,array=hid_best),
  fits.Column(name='offset', format='D' ,array=offset_list),
  fits.Column(name='mass_best', format='D',array=mass_best),
  fits.Column(name='lambda', format='D' ,array=lam),
  fits.Column(name='rlambda', format='D' ,array=rlam),

]
coldefs = fits.ColDefs(cols)
tbhdu = fits.BinTableHDU.from_columns(coldefs)
tbhdu.writeto(f'{output_loc}/model_redmagic/richness_d90_{rank_proxy}_best_match.fit', overwrite=True)
'''

# Append the best-matched halos to a richness file
with fits.open(richness_fname) as hdul:
    data = hdul[1].data
    data = data[sel_rich]
    table = Table(data)
    table['hid_host_cen'] = hid_host_cen
    table['hid_best_match'] = hid_best
    table['mass_best_match'] = mass_best
    table['offset'] = offset_list

    table['cluster_id_sanity'] = cluster_id
    new_hdu = fits.BinTableHDU(table)
    new_hdul = fits.HDUList(hdul[0:1] + [new_hdu] + hdul[2:])
    new_hdul.writeto(f'{output_loc}/model_redmagic/richness_{rank_proxy}_best_match.fit', overwrite=True)


#### sanity check using hid_host_cen
data_test = fitsio.read(f'{output_loc}/model_redmagic/richness_{rank_proxy}_best_match.fit')
id1 = data_test['hid_host_cen']
id2 = data_test['hid_host']
print('hid matched?', np.isclose(max(abs(id1-id2)), 0)) # should be zero

id1 = data_test['cluster_id']
id2 = data_test['cluster_id_sanity']
print('cluster id matched?', np.isclose(max(abs(id1-id2)), 0)) # should be zero