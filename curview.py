#!/usr/bin/env python3

from glob import glob
import pickle

import os.path

import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from skimage.morphology import remove_small_holes, remove_small_objects, label
from skimage.morphology import binary_opening, binary_erosion
from skimage.morphology import square, binary_closing, diamond, disk

from skimage.exposure import rescale_intensity

pickles = glob('clahe_curvatures/*.pickle')

# combine all candidates (unsafely specify exact size :< )
cumulative = np.zeros((2200,2561))
cumulative_all = np.zeros((2200,2561))

for n, pick in enumerate(sorted(pickles), 1):

    with open(pick, 'rb') as f:
        p = pickle.load(f)
    
    K1 = p['K1']
    K2 = p['K2']
    sigma = p['sigma']
    
    if sigma != int(sigma):
        print('skipping σ={}'.format(sigma))
        continue

    # get 'structureness' for each channel at each pixel
    R = (K1 / K2)**2 
    
    # conservative threshold
    R_thresh = ma.median(R)

    targets = (R < R_thresh).all(axis=-1).filled(0)

    # erode based on scale space
    selem = disk(sigma)
    targets = binary_opening(targets, selem=selem)
    
    #targets = remove_small_objects(targets, min_size=selem.sum() + 1)
    labeled, n_labels = label(targets, background=0, connectivity=1, return_num=True)
   
    sizes = np.bincount(labeled.flatten())
    rankings = np.argsort(sizes)

    # labels for the N largest regions
    N = 30 # number of regions to view
    largest = rankings[-(1+N):-1]
    
    #size_thresh = sigma**2
    #largest = np.argwhere(sizes >= size_thresh).flatten()[1:]
    #N= largest.size
    #if N > 100:
    #    print('{} is too many. first 100 only.'.format(N))
    #    largest = largest[:100]
    #    N = 100
    candidates = np.zeros_like(labeled)

    for region_label in largest:
        candidates = np.logical_or(candidates, labeled == region_label)

    
    targets[targets != 0] = .1

    c = candidates.astype('float') + targets

    print('figure({}):'.format(n),
            'for σ={},'.format(sigma),
          'there are {} 4-connected components.'.format(n_labels),
          'showing largest {}.'.format(N))

    #fig = plt.figure(n)
    #plt.imshow(candidates, cmap=plt.cm.Blues)
    #plt.imshow(c, cmap=plt.cm.Blues)
    #plt.title('$\sigma={}$'.format(sigma))

    #plt.axis('off')

    #fig.tight_layout()
    #fig.savefig('fig%02d.png' % n)
    #plt.show(block=False)
    
    savefile = ''.join(('%04d' % (sigma*100), '_%02d-all-se.png' % N))
    savefile = os.path.join('rb_segs/', savefile)
    
    plt.imsave(savefile, c, cmap=plt.cm.Blues)

    cumulative += candidates
    cumulative_all += (targets != 0).astype('uint8')

plt.imshow(rescale_intensity(cumulative), cmap=plt.cm.Blues)
