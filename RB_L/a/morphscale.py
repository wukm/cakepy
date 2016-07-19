#!/usr/bin/env python3

from glob import glob
import pickle

import os.path

import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import square, diamond, disk

from skimage.exposure import rescale_intensity

MODE = 'L' # or 'RGB'
OUTPUT_DIR = 'RB_L/'


pickles = glob('clahe_curvatures/*-{}.pickle'.format(MODE))

# combine all candidates (unsafely specify exact size :< )
cumulative = np.zeros((2200,2561))
cumulative_all = np.zeros((2200,2561))

for n, pick in enumerate(sorted(pickles), 1):

    with open(pick, 'rb') as f:
        p = pickle.load(f)
    
    K1 = p['K1']
    K2 = p['K2']
    sigma = p['sigma']
    
    # get 'structureness' for each channel at each pixel
    R = (K1 / K2)**2 
    
    # conservative threshold
    R_thresh = ma.median(R)
    
    targets = (R < R_thresh).filled(0)

    # erode based on scale space
    erode_selem = disk(np.round(np.sqrt(2)*sigma))
    dilate_selem = disk(sigma)
    min_size = dilate_selem.sum()*2
    targets = binary_erosion(targets, selem=erode_selem)
    if sigma < 2:
        targets = remove_small_objects(targets, min_size=10)
    targets = binary_dilation(targets, selem=dilate_selem)

    labeled, n_labels = label(targets, background=0,
                                connectivity=1, return_num=True)
   
    sizes = np.bincount(labeled.flatten())
    rankings = np.argwhere(sizes >= min_size)

    # labels for the N largest regions
    N = rankings.size - 1 # number of regions to view
    largest = rankings[1:] # everything except background
    
    candidates = np.zeros_like(labeled)

    for region_label in largest:
        candidates = np.logical_or(candidates, labeled == region_label)

    
    targets[targets != 0] = .1

    c = candidates.astype('float') + targets

    print('figure({}):'.format(n),
            'for σ={},'.format(sigma),
          'there are {} 4-connected components.'.format(n_labels),
          'showing largest {}.'.format(N))

    savefile = ''.join(('%04d' % (sigma*100), '-se.png'))
    savefile = os.path.join('RB_L/', savefile)
    
    plt.imsave(savefile, c, cmap=plt.cm.Blues)

    cumulative += candidates
    cumulative_all += (targets != 0).astype('uint8')

plt.imsave(os.path.join(OUTPUT_DIR, 'cumulative.png'),
            cumulative, cmap=plt.cm.Blues) 

plt.imsave(os.path.join(OUTPUT_DIR, 'cumulative_binary.png'),
            cumulative != 0, cmap=plt.cm.Blues) 

plt.imsave(os.path.join(OUTPUT_DIR, 'cumulative_all.png'),
            cumulative_all,  cmap=plt.cm.Blues) 

plt.imsave(os.path.join(OUTPUT_DIR, 'cumulative_all_binary.png'),
            cumulative_all != 0,  cmap=plt.cm.Blues) 

