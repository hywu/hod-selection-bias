#!/usr/bin/env python
import h5py as h5
import numpy as np

def merge_richness_files(out_path, ofname_base, boxsize, quad_file=False):    
    Nslices  = int(boxsize / 100.0) #Assumes boxsize is a multiple of 100.0 Mpc/h
    outfile = out_path + f'/{ofname_base}.h5' #+ run_name
    print(outfile)

    dummy_array = np.array([[], [], [], [], [], [], []]) #gid mass px py pz rlam lam


    i = 0
    while i < Nslices:
        j = 0
        while j < Nslices:
            fname = out_path+f"/{ofname_base}_pz0_"+str(int(boxsize))+"_px"+str(int(i*boxsize/Nslices))+"_"+str(int((i+1)*boxsize/Nslices))+"_py"+str(int(j*boxsize/Nslices))+"_"+str(int((j+1)*boxsize/Nslices))+".dat"
            dummy = np.genfromtxt(fname)
            dummy_array = np.concatenate( (dummy_array, np.transpose(dummy)), axis=1)
            j += 1
        i += 1

    N_out = len(dummy_array[0])

    out_dtype  = np.dtype([("gid",      np.int32,   1),
                           ("mass",     np.float32, 1),
                           ("x",        np.float32, 1),
                           ("y",        np.float32, 1),
                           ("z",        np.float32, 1),
                           ("R_lambda", np.float32, 1),
                           ("lambda",   np.float32, 1)])

    out_array = np.empty((N_out,), dtype=out_dtype)

    out_array['gid']      = dummy_array[0]
    out_array['mass']     = dummy_array[1]
    out_array['x']        = dummy_array[2]
    out_array['y']        = dummy_array[3]
    out_array['z']        = dummy_array[4]
    out_array['R_lambda'] = dummy_array[5]
    out_array['lambda']   = dummy_array[6]

    outfile = h5.File(outfile, 'w')
    outfile.create_dataset('halos', (N_out,), dtype=out_dtype, data=out_array, chunks=True, compression="gzip") #halos or clusters for dataset name
    outfile.close()



if __name__ == '__main__':

    import os, glob

    for phase in [0]:#range(0, 20):
        loc_out = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/' # output location

        loc_in = f'/bsuhome/hwu/scratch/Projection_Effects/Catalogs/fiducial-{phase}/z0p3/' # data location
        os.chdir(loc_in)
        hod_list = glob.glob('memHOD*z0p3.hdf5')

        os.chdir('/bsuhome/hwu/work/hod-selection-bias/repo/richness') # current location
        print('phase', phase)

        #hod_list = ['memHOD_11.2_12.4_0.65_1.0_0.2_0.0_0_z0p3_noperc.hdf5']
        hod_list = [f'memHOD_11.2_12.4_0.65_1.0_0.2_0.0_{phase}_z0p3.hdf5'] # fid
        for hod in hod_list:
            hod = hod[:-5]
            if quad_file==False:
                ofname = loc_out + hod + '/'+ hod + f'.richness_d{depth}_r{radius}.hdf5'
            elif quad_file==True:
                ofname = loc_out + hod + '/'+ hod + f'.richness_quad_d{depth}_r{radius}.hdf5'
                
            #print(ofname)
            if os.path.exists(ofname):
                print('done '+hod)
            else:
                print('doing '+hod)
                merge_richness_files(phase, hod)
