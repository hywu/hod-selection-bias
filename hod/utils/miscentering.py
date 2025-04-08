#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.special import gdtr
from scipy.interpolate import interp1d

rng = default_rng(42)

class Miscentering(object):
    def __init__(self, f_miscen, tau):
        self.f_miscen = f_miscen
        x = np.linspace(0,2,200)
        #tau = 0.165
        a = 1/tau
        b = 2
        cdf = gdtr(a,b,x)
        self.cdf_interp = interp1d(cdf, x)
        self.cdf_max = max(cdf)
        
    def draw_miscen_pos(self, x_true, y_true, R_lam):
        n = len(x_true)
        ## first decide whether a halo is miscentered
        is_miscen = (rng.random(n) < self.f_miscen)
        x_output = x_true * 1.
        y_output = y_true * 1.
        
        ## draw the miscentered 
        n_miscen = len(x_true[is_miscen])
        print('miscen frac', n_miscen / n)
        theta = rng.random(n_miscen) * 2 * np.pi
        draw = rng.random(n_miscen)
        draw[draw > self.cdf_max] = self.cdf_max # avoid the occasional interpolation problem
        r_rand = self.cdf_interp(draw) * R_lam[is_miscen]
        
        ## assemble
        x_output[is_miscen] = x_true[is_miscen] + r_rand * np.cos(theta)
        y_output[is_miscen] = y_true[is_miscen] + r_rand * np.sin(theta)  
        return x_output, y_output

if __name__ == "__main__":
    miscen = Miscentering(f_miscen=0.165, tau=0.165)
    # fake data 
    x_true, y_true = np.meshgrid(np.arange(0, 10, 1), np.arange(0, 10, 1))
    x_true = np.concatenate(x_true)
    y_true = np.concatenate(y_true)
    R_lam = 1 + np.zeros(len(x_true))
    
    x_mis, y_mis = miscen.draw_miscen_pos(x_true, y_true, R_lam)
    plt.scatter(x_true, y_true)
    plt.scatter(x_mis, y_mis, marker='x')
    #plt.savefig('test.png')