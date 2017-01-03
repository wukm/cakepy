#!/usr/bin/env python3

import numpy.ma as ma
import numpy as np

from get_base import *
from curvemap import principal_curvatures

import pickle
import os.path
import os

#MODE = 'G'  # needs to be 'L' or 'RGB' or 'G'
MODES = ['G']

SIGMAS = list(range(1,13))
SIGMAS.append(0.1)
DATADIR = 'clahe_curvatures/TA-BN0657337'
if os.path.exists(DATADIR):
    pass
else:
    os.mkdir(DATADIR)

FILENAME = 'samples/traces/TA-BN0657337.png'

#clahe_pickle = 'clahe-' + mode + '.npy'
#    try:
#        I = np.load(clahe_pickle)
#    except FileNotFoundError:
#        I = get_preprocessed(mode=mode)
#        np.save(clahe_pickle, I)

for MODE in MODES:

    I = get_preprocessed(FILENAME, mode=MODE)

    for sigma in SIGMAS:
        
        filename = ''.join(("K-%04d-" % (sigma*100), MODE, '.pickle'))
        
        filename = os.path.join(DATADIR, filename)

        if os.path.isfile(filename):
            print('curvatures for σ={}'.format(sigma),
                'have already been generated. skipping.')
            continue
        else:
            print('starting work with σ={}'.format(sigma)) 

        K1, K2 = principal_curvatures(I, sigma=sigma)
        
        p = {   'sigma' : sigma,
                'K1' : K1,
                'K2' : K2}

        with open(filename, 'wb') as f:
            pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)

        print('pickle for σ={} saved successfully!'.format(sigma))
        print('(using filename: ', filename)




        
