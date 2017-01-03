#!/usr/bin/env python3

import numpy.ma as ma
import numpy as np

from get_base import get_preprocessed
from curvemap import principal_directions

import pickle
import os.path

import datetime

import matplotlib.pyplot as plt

MODE = 'G'
SIGMAS = list(range(1,13))
SIGMAS.append(0.1)
DATADIR = 'clahe_directions/TA-BN0657337'

if os.path.exists(DATADIR):
    pass
else:
    os.mkdir(DATADIR)
FILENAME = 'samples/traces/TA-BN0657337.png'

print('pickling principal directions for mode={}'.format(MODE),
        'at the following scales:')
print('\tσ=', SIGMAS)
print('\tand saving in directory:', './'+DATADIR)
print()

I = get_preprocessed(FILENAME, mode=MODE)

for sigma in SIGMAS:
    
    filename =  ''.join(('T-%04d-' % (sigma*100), MODE, '.pickle'))
    filename =  os.path.join(DATADIR, filename)  

    if os.path.isfile(filename):
        print('principal directions for σ={}'.format(sigma),
                'have already been generated. skipping.')
        continue
    else:
        print('* * * ', 'starting work with σ={}'.format(sigma), '* * *')
    
    # lol what is a decorator, lol what is ipython
    before = datetime.datetime.now()
    T,L = principal_directions(I, sigma=sigma)
    after = datetime.datetime.now()
    print('>>> elapsed time for σ={} was: '.format(sigma), after-before)

    p = {   'sigma' : sigma,
            'T' : T,
            'L' : L}

    with open(filename, 'wb') as f:
        pickle.dump(p, f, pickle.HIGHEST_PROTOCOL)
    

    print('... pickle for σ={} saved successfully!'.format(sigma))
    print('... with filename: ', filename)

    print('... saving image files...', end=' ')
    image_file = os.path.join(DATADIR, '%04d-L.png' % (sigma*100))
    plt.imsave(image_file, L.filled(0), cmap=plt.cm.spectral)

    image_file = os.path.join(DATADIR, '%04d-T.png' % (sigma*100))
    plt.imsave(image_file, T.filled(0), cmap=plt.cm.spectral)
    print('done')
