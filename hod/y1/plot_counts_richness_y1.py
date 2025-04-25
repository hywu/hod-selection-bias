#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_counts_richness_y1(iz, axes=None):
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    out_loc = os.path.join(BASE_DIR, 'data')

    z_list = [0.2, 0.35, 0.5, 0.65]
    zmin = z_list[iz]
    zmax = z_list[iz+1]
    lam_min_list, den_list, den_low, den_high = np.loadtxt(out_loc+f'/des_y1_space_density_lambda_z_{zmin}_{zmax}.dat', unpack=True)
    plt.plot(lam_min_list, den_list, label='DES Y1', c='k', marker='o')
    plt.fill_between(lam_min_list, den_low, den_high, facecolor='gray', alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$n(>\lambda)$')
    plt.legend()
    plt.xlim(10, None)