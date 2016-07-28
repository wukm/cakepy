#!/usr/bin/env python3

import os
import os.path
from sys import exit

import pickle

import matplotlib.pyplot as plt

import numpy as np
import numpy.ma as ma

from skimage.morphology import *
from skimage.transform import rotate

from skimage.exposure import rescale_intensity

from skimage.color import label2rgb
from functools import partial

def load_principal_data(sigma, mode):
    """
    load principal curvature and principal direction matrices for a
    given scale space from an existing pickle. Does not do
    any error handling currently, so please make sure that the file exists first.
    Please read source code for file name spec (this behavior will obviously be
    fixed in the future)
    
    INPUT:

    sigma:  the scale space as a number
    mode:   one of 'G', 'L', or 'RGB'
    
    OUTPUT:

    (K1, K2): matrices of principal curvatures
    """

    curve_datadir = 'clahe_curvatures'
    theta_datadir = 'clahe_directions'
        
    pickle_file = 'K-%04d-' % (sigma*100)

    pickle_file = ''.join((pickle_file, mode, '.pickle'))
    pickle_file = os.path.join(curve_datadir, pickle_file)
    
    with open(pickle_file, 'rb') as f:
        p = pickle.load(f)

    K1 = p['K1']
    K2 = p['K2']
    s = p['sigma']

    pickle_file = 'T-%04d-' % (sigma*100)
    pickle_file = ''.join((pickle_file, mode, '.pickle'))
    pickle_file = os.path.join(theta_datadir, pickle_file)

    with open(pickle_file, 'rb') as f:
        p = pickle.load(f)

    L = p['L']
    T = p['L']

    return K1, K2, T, L



def vessel_filter(img, thetas, sigma, length_ratio=4, verbose=True):
    """
    runs a curvilinear filter at the given scale space `sigma`
    
    INPUT:
        img:            a binary 2D array
        sigma:          the scale space
        length_ratio:   a rectangular filter will be applied with size 
        steps:          the range of rotations (0,180) is divided into this
                        many steps (default: 16 or 12 degrees)
        verbose:        default True

    OUTPUT:
        extracted:  a 2D binary array of the same shape as img
    
    METHODS:
        (todo)

    IMPLEMENTATION:
        (todo)  

    WARNINGS/BUGS:
        this may be supremely wasteful for large step sizes. you should check
        in the anticipated range of sigmas that there is sufficient variation
        in the rotated structure elements to warrant that amount of step sizes.
        print('cleaning up scale space')

        furthermore, this filter should be used carefully. there are probably
        bugs in the logic and implementation.
    """
    extracted = np.zeros_like(img)

    width, length = int(2*sigma), int(2*sigma*length_ratio)

    rect = rectangle(width, length)
    thetas = np.round(thetas*180 / np.pi)
    thetas[thetas==180] = 0

    if verbose:
        print('running vessel_filter with σ={}: w={}, l={}'.format(
            sigma, width,length), end='\t', flush=True)

    #if verbose:
    #    print('θ', end=' ', flush=True)

    if verbose:
        print('building rotated filters...', end=' ')
    srot = partial(rotate, resize=True, preserve_range=True) # look at order
    rotated = [srot(rect, theta) for theta in range(180)]
    
    if verbose:
        print('done.')

    for theta in range(180):
        if verbose:
            print('θ=', theta, end='\t', flush=True)
            if theta % 6 == 0:
                print()

        vessels = binary_erosion(img, selem=rotated[theta])
        #vessels = binary_opening(img, selem=rotated[theta])
        extracted = np.logical_or(extracted, (thetas == theta) * vessels) 
    if verbose:
        print('') # new line
    
    return extracted

def get_targets(K1,K2):
    """
    calculate the frangi 'blobness' measure R for the given principal
    curvatures. return a binary filter with the conservative threshold
    of R < R_median
    """ 
    #R = (K1 / K2) ** 2
    R = (K1**2 + K2**2)/2

    return (R > ma.median(R)).filled(0)

if __name__ == "__main__":

    b = partial(plt.imshow, cmap=plt.cm.Blues)
    s = plt.show


    mode = 'G'

    # make true if you want to restrict each filter to unextracted targets only 
    exclusivity = False

    scale_range = list(range(19,1,-1))

    length_ratio = .5

    cumulative = np.zeros((2200,2561), dtype='uint8')
    extracted_all = np.zeros((2200,2561,len(scale_range)), dtype='uint8') 
    OUTPUT_DIR = 'scvme_output'
    SUBDIR = '_'.join((mode, 'directed',
                        'lr={}'.format(length_ratio))
                        )
    if exclusivity:
        SUBDIR = '_'.join((SUBDIR, 'exclusive'))
    
    print('saving outputs in', os.path.join(OUTPUT_DIR, SUBDIR))

    try:
        os.mkdir(os.path.join(OUTPUT_DIR, SUBDIR))
    except FileExistsError:
        ans = input('save path already exists! would you like to continue? [y/N]')
        if ans != 'y':
            print('aborting program. clean up after yourself.')

            exit(0)

        else:
            print('your files will be overwritten (but artifacts may remain!)')
    finally:
        print('\n')
   
    for n, sigma in enumerate(scale_range):

        K1, K2, L, T = load_principal_data(sigma, mode)
        
        targets = get_targets(K1,K2)
        
        savefile = ''.join(('%02d' % sigma, '-raw.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile), targets*L,
                cmap=plt.cm.spectral)

        if exclusivity:
            # remove any previously extracted vessels from consideration
            targets[cumulative.nonzero()] = False

        extracted = vessel_filter(targets, L, sigma, length_ratio)
        extracted_all[:,:,n] = extracted.filled(0)
        savefile = ''.join(('%02d' % sigma, '.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile),
                extracted.filled(0), cmap=plt.cm.Blues)
        
        # label regions with this sigma in the cumulative map, but only where no label exists
        new_labels = sigma*np.logical_and(extracted != 0, cumulative == 0).filled(0)
        cumulative += new_labels.astype('uint8')
          

    print('building labeled/cumulative map')

    c_img = label2rgb(cumulative, bg_label=0)
    
    bins = np.unique(c_img)
        
    #    # make a label set
    #    #cols = label2rgb(bins, bg_label=0)
    #    #cols = np.repeat(cols, 20, axis=0)
    #    #np.tile(np.expand_dims(cols, axis=1), (1,30,1))

    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative.png'), c_img)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative_binary.png'),
                                cumulative!=0, cmap=plt.cm.Blues)
