#!/usr/bin/env python
import numpy as np


def periodic_boundary_condition(x, y, z, boxsize, x_padding, y_padding, z_padding, *args):
    # takes arbitrary number of extra properties
    # output the same number of extra properties

    if (boxsize - np.max(x)) > 0.01 * boxsize:
        print('boxsize is probably wrong!')

    prop_dict = {'x':[], 'y':[], 'z':[]}
    #print('np.shape(args)', np.shape(args))
    nprop = np.shape(args)[0]
    print('periodic boundary condition dealing with ' + str(nprop) + ' extra properties')

    for iprop in range(nprop):
        prop_dict[f'p{iprop}'] = []

    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                x_temp = x + i * boxsize
                y_temp = y + j * boxsize
                z_temp = z + k * boxsize
                selx = (x_temp > - x_padding)&(x_temp < boxsize + x_padding)
                sely = (y_temp > - y_padding)&(y_temp < boxsize + y_padding)
                selz = (z_temp > - z_padding)&(z_temp < boxsize + z_padding)
                sel = selx & sely & selz
                prop_dict['x'].extend(x_temp[sel])
                prop_dict['y'].extend(y_temp[sel])
                prop_dict['z'].extend(z_temp[sel])

                for iprop in range(nprop):
                    prop_dict[f'p{iprop}'].extend(args[iprop][sel])

    return [np.array(prop_dict[key]) for key in prop_dict.keys()]


