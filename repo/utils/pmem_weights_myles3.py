import numpy as np
from scipy import spatial

#calling everything that is a z-score a x-score because z meaning 2 things is confusing

data_in = np.loadtxt('data/myles_pmem_sampling.dat')

delz = abs(data_in[:,0])
RspecovRlambda = data_in[:,1]
pmem = data_in[:,2]

delz_mean = np.mean(delz)
RspecovRlambda_mean = np.mean(RspecovRlambda)

delz_std = np.std(delz)
RspecovRlambda_std = np.std(RspecovRlambda)

def x_score(data, mean, std):
	return (data - mean) / std

delz_x = x_score(delz, delz_mean, delz_std)
RspecovRlambda_x = x_score(RspecovRlambda, RspecovRlambda_mean, RspecovRlambda_std)

data_x_tree = spatial.cKDTree(np.dstack([delz_x, RspecovRlambda_x])[0])

def pmem_weights(dz_input, R_input, dz_max):
	pmem_out = np.zeros(np.shape(dz_input))

	#selecting points within dz_max
	sel_dz_max = (abs(dz_input) < dz_max)
	dz_input_sel = dz_input[sel_dz_max]
	R_input_sel = R_input[sel_dz_max]
	pmem_out_sel = pmem_out[sel_dz_max]

	dz_input_x = x_score(abs(dz_input_sel), delz_mean, delz_std)
	R_input_x = x_score(R_input_sel, RspecovRlambda_mean, RspecovRlambda_std)

	input_x = np.dstack([dz_input_x, R_input_x])[0]

	distances, indices = data_x_tree.query(input_x, k=1, distance_upper_bound=0.01)

	sel = (distances < np.inf)
	pmem_out_sel[sel] = pmem[indices[sel]]
	pmem_out[sel_dz_max] = pmem_out_sel
	return pmem_out
	
if __name__=='__main__':
	test_dz = np.linspace(-0.1,0.1,20)
	test_R = np.linspace(-0.01,1.01,20)
	test_pmem = pmem_weights(test_dz, test_R, 0.05)
	for i in range(len(test_dz)):
		print(f"dz:{test_dz[i]:.3f} R:{test_R[i]:.3f} pmem:{test_pmem[i]:.3f}")


