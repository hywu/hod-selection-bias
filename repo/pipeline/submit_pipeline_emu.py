#!/usr/bin/env python
import os
import numpy as np

fname = f'sbatch_output/gals.sbatch'

script = f'''#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=halos
#SBATCH --array=4-20
#SBATCH --output=sbatch_output/halos_%A_%a.out
#SBATCH --account=hywu_cluster_sims_0001
#SBATCH --partition=standard-s
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB 
./pipeline_emu.py $SLURM_ARRAY_TASK_ID 2>&1
'''
f = open(fname, 'w')
f.write(script)
f.close()
os.system('sbatch '+fname)