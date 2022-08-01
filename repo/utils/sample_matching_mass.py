#!/usr/bin/env python
import numpy as np

def sample_matching_mass(lnM_select, lnM_all, x_all, y_all, z_all, dm=0.1, factor=5):
    #### set up the bins for mass
    min_m = min(lnM_all)-dm
    max_m = max(lnM_all)+dm
    m_bins = np.arange(min_m, max_m+dm, dm)
    nM = len(m_bins)-1

    x_matched = []
    y_matched = []
    z_matched = []
    lnM_matched = []
    
    for iM in range(nM):
        m_lo = m_bins[iM]
        m_hi = m_bins[iM+1]

        select_bin = (lnM_select >= m_lo)&(lnM_select < m_hi)
        n_to_draw = len(lnM_select[select_bin])

        select_all = (lnM_all >= m_lo)&(lnM_all < m_hi)
        n_this_bin = len(lnM_all[select_all])

        if n_to_draw > 0:
            n_to_draw *= factor
            #print('n_this_bin vs. n_to_draw', n_this_bin, n_to_draw)
            select_rand = np.random.choice(n_this_bin, n_to_draw, replace=True)
            x_matched.extend(x_all[select_all][select_rand])
            y_matched.extend(y_all[select_all][select_rand])
            z_matched.extend(z_all[select_all][select_rand])
            lnM_matched.extend(lnM_all[select_all][select_rand])

    x_matched = np.array(x_matched).flatten()
    y_matched = np.array(y_matched).flatten() 
    z_matched = np.array(z_matched).flatten() 
    lnM_matched = np.array(lnM_matched).flatten() 
    
    return x_matched, y_matched, z_matched, lnM_matched

