import numpy as np

def pmem_weights_dchi(dchi_input, R_input, dchi_max, depth):
    pmem_out = np.zeros(np.shape(dchi_input))
    sel = (abs(dchi_input) < depth) & (abs(dchi_input) < dchi_max) & (R_input < 1.0)
    pmem_out[sel] = 1.0 - (dchi_input[sel]/depth)**2
    return pmem_out

def volume_dchi(R, depth): # Integrate[Pi f[x]^2, {x, -q, q}]
    return (8./15.) * np.pi * R**2 * (2*depth)