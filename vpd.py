#!/usr/bin/env python3

from glob import glob
import pickle

import os.path

import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from functools import partial

from scipy.stats import circmean, circstd, circvar
from scipy.ndimage import generic_filter
from skimage.morphology import disk

import datetime

MODE = 'G' # 'RGB' or 'G' or 'L'
OUTPUT_DIR = 'vpd-G'
SAVE = False

pickles = glob('clahe_directions/*-{}.pickle'.format(MODE))

# combine all candidates (unsafely specify exact size :< )
# cumulative = np.zeros((2200,2561))
# cumulative_all = np.zeros((2200,2561))

cv = partial(circvar, low=0, high=np.pi)

for n, pick in enumerate(sorted(pickles), 1):
    
    with open(pick, 'rb') as f:
        p = pickle.load(f)

    T = p['T']
    L = p['L']
    sigma = p['sigma']
    
    if sigma != 3:
        continue

    print('σ={}'.format(sigma))
    
    before = datetime.datetime.now() 
    print('calculating local (circular) variance for T:= θ_1')
    T_var = generic_filter(T, cv, footprint=disk(sigma))
    print('calculating local (circular) variance for L:= θ_2')
    L_var = generic_filter(L, cv, footprint=disk(sigma))
    after = datetime.datetime.now() 
    
    print('total time: ', after - before)

    T_var = ma.masked_array(T_var, mask=T.mask)
    L_var = ma.masked_array(L_var, mask=L.mask)
    
    T_thresh = ma.median(T_var)
    L_thresh = ma.median(L_var)
    
    print('T_thresh =', T_thresh)
    print('L_thresh =', L_thresh)
    filestub = '-'.join(('%04d' % (sigma*100), '{}', MODE))
    
    if SAVE:
        plt.imsave(os.path.join(OUTPUT_DIR, ''.join((
            filestub.format('T'), '.png'))),
            (T_var < T_thresh).filled(0), cmap=plt.cm.Blues) 

        plt.imsave(os.path.join(OUTPUT_DIR, ''.join((
            filestub.format('L'), '.png'))),
            (L_var < L_thresh).filled(0), cmap=plt.cm.Blues) 
    
        print('images saved')

    p_out = { 'sigma' : sigma,
            'T_var' : T_var,
            'L_var' : L_var}
    
    # should check file first
    if SAVE:
        print('saving pickle...', end=' ')

        pickle_out = os.path.join(OUTPUT_DIR,
                ''.join((filestub.format('var'), '.pickle')))

        with open(pickle_out, 'wb') as f:
            pickle.dump(p_out, f, pickle.HIGHEST_PROTOCOL) 

    print('done.')

