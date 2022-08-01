#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import poisson

def Ngal_S20_noscatt_noround(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    y = max(0, y)
    Nsat = Ncen * (y ** alpha)
    return (Ncen + Nsat)

def Ngal_S20_noscatt(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    y = max(0, y)
    Nsat = Ncen * (y ** alpha)
    return (np.round(Ncen + Nsat)+1e-4).astype(int)

def Ngal_S20_poisson(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    y = max(0, y)
    Nsat_mean = Ncen * (y ** alpha)
    Nsat_mean = (np.round(Nsat_mean)+1e-4).astype(int)
    Nsat = poisson.rvs(Nsat_mean)
    return (np.round(Ncen + Nsat)+1e-4).astype(int)

def Ngal_S20_gauss(M, alpha=1., lgM1=12.9, kappa=1., lgMcut = 11.7, sigmalogM=0.1, sigma_intr=0.2):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    sigmalogM = 0.1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    y = max(0, y)
    Nsat_mean = Ncen * (y ** alpha)
    Nsat_mean = np.random.normal(loc=Nsat_mean, scale=Nsat_mean*sigma_intr)
    Nsat_mean = (np.round(Nsat_mean)+1e-4).astype(int)
    if Nsat_mean >=0:
        Nsat = poisson.rvs(Nsat_mean)
    else:
        Nsat = 0
    return (np.round(Ncen + Nsat)+1e-4).astype(int)


if __name__ == "__main__":
    M = 1e12
    print(Ngal_S20_noscatt(M))
    print(Ngal_S20_poisson(M))
    print(Ngal_S20_gauss(M, alpha=1.3))