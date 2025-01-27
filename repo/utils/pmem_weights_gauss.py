import numpy as np

def pmem_weights_dchi(dchi_input, R_input, dchi_max, depth):
    pmem_out = np.zeros(np.shape(dchi_input))
    sel = (abs(dchi_input) < dchi_max) & (R_input < 1.0)
    pmem_out[sel] = np.exp(-0.5 * dchi_input[sel]**2 / depth**2)
    return pmem_out

def volume_dchi(R, depth): # Integrate[Pi f[x]^2, {x, -Infinity, Infinity}, Assumptions -> sig > 0]
    return 0.5 * (np.pi**1.5) * R**2 * (2.*depth)