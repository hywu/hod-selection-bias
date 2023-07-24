import numpy as np
from kllr import kllr_model
import fitsio
import sys

loc_fname = sys.argv[1]
index = loc_fname.index('/')
cat_name = loc_fname[index+10:-4]
loc = loc_fname[:index+1]

data, header = fitsio.read(loc_fname, header=True)

mass = data['M200m']
sel = (mass > 5e12)
mass = mass[sel]
richness = data['lambda'][sel]


def calc_kllr():
	kllr_results = []

	lm = kllr_model(kernel_type='gaussian', kernel_width = 0.2)
	temp = lm.fit(np.log(mass), richness, bins=25)

	kllr_results.append(np.exp(temp[0]))
	kllr_results.append(np.mean(temp[1], axis=0))
	kllr_results.append(np.mean(temp[4], axis=0))
	kllr_results.append(np.std(temp[4], axis=0))
	return kllr_results

if __name__=='__main__':
	kllr_results = calc_kllr()
	np.savetxt(loc+'kllr_' + cat_name + '.dat', np.transpose(kllr_results), '%g', header='mass, richness, richness_err, richness_err_err')