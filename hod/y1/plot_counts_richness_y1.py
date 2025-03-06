#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_counts_richness_y1(axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_loc = os.path.join(BASE_DIR, 'data')

    lam_min_list, den_list, den_low, den_high = np.loadtxt(out_loc+'/des_y1_space_density_lambda_z_0.2_0.35.dat', unpack=True)
    plt.plot(lam_min_list, den_list, label='DES Y1', c='k', marker='o')
    plt.fill_between(lam_min_list, den_low, den_high, facecolor='gray', alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$n(>\lambda)$')
    plt.legend()
    plt.xlim(10, None)