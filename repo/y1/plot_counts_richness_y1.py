#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def plot_counts_richness_y1(axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    lam_min_list, den_list, den_low, den_high = np.loadtxt('data/des_y1_space_density_lambda_z_0.2_0.35.dat', unpack=True)
    plt.plot(lam_min_list, den_list, label='DES Y1', c='k', marker='o')
    plt.fill_between(lam_min_list, den_low, den_high, facecolor='gray', alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$n(>\lambda)$')
    plt.legend()
    plt.xlim(10, None)