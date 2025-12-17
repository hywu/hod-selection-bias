#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.stats import poisson
from scipy.stats import bernoulli
from scipy.stats import norm

def Ngal_S20_noscatt(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
    M = np.atleast_1d(M)
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen = 0.5 * (1 + special.erf(x))
    y = (M - kappa * Mcut) / M1
    y = np.maximum(y, 0)
    Nsat = Ncen * (y ** alpha)
    #Nsat = max(0, Nsat)
    return Ncen, Nsat

## TODO: merge with the 
# def Ngal_S20_noscatt_arr(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1):
#     M = np.atleast_1d(M)
#     Mcut = 10**lgMcut
#     M1 = 10**lgM1
#     x = (np.log10(M)-np.log10(Mcut))/sigmalogM
#     Ncen = 0.5 * (1 + special.erf(x))
#     y = (M - kappa * Mcut) / M1
#     y[y < 0] = 0
#     Nsat = Ncen * (y ** alpha)
#     return Ncen, Nsat


def Ngal_S20_poisson(M, alpha=1., lgM1=12.9, kappa=1., lgMcut=11.7, sigmalogM=0.1, sigmaintr=0, fcen=1.):
    Mcut = 10**lgMcut
    M1 = 10**lgM1
    x = (np.log10(M)-np.log10(Mcut))/sigmalogM
    Ncen_mean = 0.5 * (1 + special.erf(x))
    Ncen = bernoulli.rvs(Ncen_mean)
    if fcen < 1.:
        Ncen_incomp = bernoulli.rvs(fcen) # not affecting Nsat
    else:
        Ncen_incomp = Ncen
    y = (M - kappa * Mcut) / M1
    y = np.maximum(y, 0)
    Nsat_mean = Ncen * (y ** alpha) # determined by Ncen, not Ncen_incomp
    #Nsat_mean = max(0, Nsat_mean)
    Nsat = poisson.rvs(Nsat_mean)
    if sigmaintr > 1e-8 and Nsat > 0:
        lnNsat = norm.rvs(np.log(Nsat), sigmaintr)
        Nsat = int(np.round(np.exp(lnNsat)))
    #print('Ncen, Nsat', Ncen, Nsat)
    return Ncen_incomp, Nsat


if __name__ == "__main__":
    for M in 10**np.linspace(11,12):
        print(Ngal_S20_poisson(M, sigmaintr=0.2))