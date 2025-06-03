#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('MNRAS')
from scipy.interpolate import interp1d
import os, sys
import h5py
import pandas as pd

# get the halo run members
# plot the red sequence
# output the redsequence model (slope, intercept, scatter)

# TODO: is it correct to use z-band?
# TODO: should i use a different pmem_cut?
# TODO: how to get 0.2 Lstar?
pmem_cut = 0.9

card_loc = '/projects/shuleic/shuleic_unify_HOD/shuleic_unify_HOD/Cardinalv3/'
fname = card_loc + 'Cardinal-3_v2.0_Y6a_mastercat.h5'
f = h5py.File(fname,'r')
print(f.keys())
data = f['catalog/redmapper_halo/lgt20/members'] # halo-run
print(data.keys())
z = data['z'][:] # is this z_cos?
p = data['p'][:]
theta_i = data['theta_i'][:]
sel_p = (p > pmem_cut)&(theta_i > 0.95)

theta_i = data['theta_i'][sel_p]
mag = data['mag'][sel_p]
z = z[sel_p]
print(np.shape(mag))
mag_g = mag[:,0]
mag_r = mag[:,1]
mag_i = mag[:,2]
mag_z = mag[:,3]


gr = mag_g - mag_r
ri = mag_r - mag_i
iz = mag_i - mag_z

color_list = np.array([gr, ri, iz])
color_name_list = ['g_r', 'r_i', 'i_z']

plot_loc = f'../../plots/cardinal_cyl/cmd_{pmem_cut}/'


#### fit the red-sequence to a straight line ####
def get_red_sequence_model(zmin, zmax):
    sel = (z > zmin)&(z < zmax)#*(p > pmem_cut)


    mag_z_limit = max(mag_z[sel]) # TODO: it's not hard cut... fit a function?

    # first plot the magnitude histogram
    plt.figure(figsize=(8,4))
    plt.subplot(121)

    #plt.hist(mag_g[sel], density=True, alpha=0.2, label='mag g')
    #plt.hist(mag_r[sel], density=True, alpha=0.2, label='mag r')
    plt.hist(mag_i[sel], density=True, alpha=0.2, label='i')
    plt.hist(mag_z[sel], density=True, alpha=0.2, label='z')
    plt.xlabel('magnitude')
    plt.legend()

    plt.subplot(122)
    plt.hist(theta_i[sel], density=True, alpha=0.2)#, label='theta i')
    plt.xlabel('theta L')
    plt.savefig(plot_loc+f'mag_z_{zmin:g}_{zmax:g}.png')

    # then color-magnitude diagram
    slope_list = []
    intercept_list = []
    std_list = []
    mean_list = []
    color_res = []
    plt.figure(figsize=(4,6))
    for ic in range(3):
        color_name = color_name_list[ic]
        color = color_list[ic]
        
        x = mag_z[sel]
        y = color[sel]

        mean_list.append(np.mean(y))

        m, b = np.polyfit(x, y, 1) # slope, intercept

        res = np.std(y - (m * x + b)) # for calcualting the covariance matrix
        slope_list.append(m)
        intercept_list.append(b)
        std_list.append(res)
    
        # plot the best-fit line
        x_plot = np.linspace(min(x), max(x))
        y_plot = m * x_plot + b
        
        #### save the figure
        plt.subplot(3,1,1+ic)
        plt.scatter(x, y, s=0.05, c='gray')
        line = plt.plot(x_plot, y_plot)
        
        plt.fill_between(x_plot, y_plot-res, y_plot+res, interpolate=True, facecolor=line[0].get_c(), alpha=0.5)
        plt.ylabel(color_name.replace('_','-'))
        #plt.ylim(0, 1)
        if ic==0: 
            #plt.legend()
            plt.title(r'Cardinal halo-run, %g<z<%g'%(zmin, zmax))#sim_name.replace('_','-')+r', $\rm M_{vir}>%.e~h^{-1}M_\odot$'%Mvir_min, fontsize=10)
        if ic==2: plt.xlabel('mag z')


        color_res.append(y - (m * x +b))



    plt.savefig(plot_loc+f'cmd_z_{zmin:g}_{zmax:g}.png')



    #### covariance matrix
    gr_res = color_res[0]
    ri_res = color_res[1]
    iz_res = color_res[2]
    
    #### correlation coefficient of the residual
    df = pd.DataFrame({'gr': gr_res, 'ri': ri_res, 'iz': iz_res})
    correlation_matrix = df.corr()
    print('correlation matrix',correlation_matrix)
    Cov = df.cov()
    #print('covariance matrix', Cov)
    Cov = np.array(Cov)
    iCov = np.linalg.inv(Cov)
    #plt.savefig(plot_loc+f'CMD_{Mvir_min:.0e}.png')
    # np.shape(data['chisq'])
    return intercept_list, slope_list, std_list, mean_list, mag_z_limit


if __name__ == "__main__":
    dz = 0.01
    zmin_list = np.arange(0.2, 0.641, dz)[0:2]

    # clear the files
    for ic in range(3):
        color_name = color_name_list[ic]
        outfile = open(f'../cardinal_cyl/data_member/{color_name}_model.dat','w')
        outfile.write('#zmin, zmax, intercept, slope, scatter, mean_color \n')
        outfile.close()
    
    outfile = open(f'../cardinal_cyl/data_member/mag_z_limit.dat','w')
    outfile.write('zmin, zmax, mag_z_limit \n')
    outfile.close()

    # save results to files
    for zmin in zmin_list:
        zmax = zmin + dz
        intercept, slope, scatter, mean, mag_z_limit = get_red_sequence_model(zmin=zmin, zmax=zmax)
        for ic in range(3):
            color_name = color_name_list[ic]
            outfile = open(f'../cardinal_cyl/data_member/{color_name}_model.dat','a')
            outfile.write('%g %g %g %g %g %g\n'
                          %(zmin, zmax, intercept[ic], slope[ic], scatter[ic], mean[ic]))
            outfile.close()

        # save the z-band limit
        outfile = open(f'../cardinal_cyl/data_member/mag_z_limit.dat','a')
        outfile.write('%g %g %g \n'%(zmin, zmax, mag_z_limit))
        outfile.close()

