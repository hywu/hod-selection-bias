#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import poisson
from scipy.stats import bernoulli

def Ngal_S20_noscatt(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    Nsat = Ncen * (y ** alpha)
    Nsat = max(0, Nsat)
    return Ncen, Nsat

def Ngal_S20_poisson(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen_mean = 0.5 * (1 + special.erf(x))
    Ncen = bernoulli.rvs(Ncen_mean)
    y = (M - kappa * Mcut) / M1
    Nsat_mean = Ncen * (y ** alpha)
    Nsat_mean = max(0, Nsat_mean)
    Nsat = poisson.rvs(Nsat_mean)
    #print('Ncen, Nsat', Ncen, Nsat)
    return Ncen, Nsat


if __name__ == "__main__":
    for M in 10**np.linspace(11,12):
        print(Ngal_S20_poisson(M))