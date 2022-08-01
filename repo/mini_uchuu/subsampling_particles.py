#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys
sys.path.append('../uchuu/')
from readGadgetSnapshot import readGadgetSnapshot
loc = '/bsuhome/hwu/scratch/uchuu/MiniUchuu/'

x = []
y = []
z = []
for ifile in range(400):
    data = readGadgetSnapshot(loc+f'snapdir_043/MiniUchuu_043.gad.{ifile}', read_pos=True)
    pos = data[1]
    x.extend(pos[:,0][::100])
    y.extend(pos[:,1][::100])
    z.extend(pos[:,2][::100])
    print(ifile)
        
cols=[
  fits.Column(name='x', format='D' ,array=x),
  fits.Column(name='y', format='D',array=y),
  fits.Column(name='z', format='D',array=z),
]
coldefs = fits.ColDefs(cols)
tbhdu = fits.BinTableHDU.from_columns(coldefs)
tbhdu.writeto(loc+'particles_043_1percent.fit', overwrite=True)
