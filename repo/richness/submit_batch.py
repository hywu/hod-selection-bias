#!/usr/bin/env python
import os, glob



def run_model(phase, model, depth):
    time = 10
    fname = f'sbatch_output/{model}.sbatch'
    script = f'''#!/bin/bash -l
#SBATCH -p bsudfq
#SBATCH --nodes=1
#SBATCH --time={time}:00:00
#SBATCH --job-name={model[7:]}
#SBATCH --output={fname}.out
##SBATCH --exclusive
./calc_richness.py {phase} {model} {depth} 2>&1 | tee {fname}.out'''
    f = open(fname, 'w')
    f.write(script)
    f.close()
    os.system('sbatch '+fname)
    #print(fname)

#SBATCH --mail-type=ALL
#SBATCH --mail-user=hywu@slac.stanford.edu
##SBATCH --mail-type=ALL
##SBATCH --mail-user=hywu@slac.stanford.edu
##SBATCH --qos regular
##SBATCH --qos debug
#run_name = 'noperc'
##SBATCH --time=00:30:00


for depth in [15]:#, 60]:
    for phase in range(20):
        loc_out = f'/bsuhome/hwu/scratch/Projection_Effects/output/richness/fiducial-{phase}/z0p3/' # output location

        loc_in = f'/bsuhome/hwu/scratch/Projection_Effects/Catalogs/fiducial-{phase}/z0p3/' # data location
        os.chdir(loc_in)
        hod_list = glob.glob('memHOD*z0p3.hdf5')

        os.chdir('/bsuhome/hwu/work/cylinder_projection/repo/richness') # current location

        hod_list = [f'memHOD_11.2_12.4_0.65_1.0_0.2_0.0_{phase}_z0p3.hdf5'] # fid
        for hod in hod_list:
            hod = hod[:-5] # get rid of .hdf5
            ofname = loc_out + hod + '/'+ hod + f'.richness_d{depth}.hdf5'
            #print(ofname)
            if os.path.exists(ofname):
                print('done '+hod)
            else:
                print('run ')
                run_model(phase, hod, depth)

