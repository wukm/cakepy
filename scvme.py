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


def load_curvatures(sigma, mode, datadir=None):
    """
    load principal curvature matrices for a given scale space
    from an existing pickle. Does not do any error handling currently,
    so please make sure that the file exists first. Please read
    source code for file name spec (this behavior will obviously
    be fixed in the future)

    INPUT:

    sigma:  the scale space as a number
    mode:   one of 'G', 'L', or 'RGB'
    datadir:    where to look for pickles (defaults to ./clahe_curvatures)
    
    OUTPUT:

    (K1, K2): matrices of principal curvatures
    """

    if datadir is None:
        datadir = 'clahe_curvatures'

    pickle_file = 'K-%04d-' % (sigma*100)
    #pickle_file = 'clahe_curvatures/K-0300-{}.pickle'.format(mode)

    pickle_file = ''.join((pickle_file, mode, '.pickle'))
    pickle_file = os.path.join(datadir, pickle_file)
    
    with open(pickle_file, 'rb') as f:
        p = pickle.load(f)

    K1 = p['K1']
    K2 = p['K2']
    s = p['sigma']

    assert np.isclose(s, sigma)

    return (K1, K2)



def vessel_filter(img, sigma, length_ratio=4, steps=16, verbose=True):
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

        furthermore, this filter should be used carefully. there are probably
        bugs in the logic and implementation.
    """
    extracted = np.zeros_like(img)

    theta_range = np.linspace(0,180,steps)

    width, length = int(2*sigma), int(2*sigma*length_ratio)

    rect = rectangle(width, length)

    if verbose:
        print('running vessel_filter with σ={}: w={}, l={}, steps={}'.format(
            sigma, width,length, steps), end='\t', flush=True)

    for theta in theta_range:
        
        if verbose:
            print('θ', end=' ', flush=True)
        angled_filter = rotate(rect, theta, resize=True, preserve_range=True)
        vessels = binary_opening(img, selem=angled_filter)
        extracted = np.logical_or(extracted, vessels)
    
    if verbose:
        print('') # new line

    return extracted

def get_targets(K1,K2):
    """
    calculate the frangi 'blobness' measure R for the given principal
    curvatures. return a binary filter with the conservative threshold
    of R < R_median
    """ 
    R = (K1 / K2) ** 2

    return (R < ma.median(R)).filled(0)

if __name__ == "__main__":

    from functools import partial

    b = partial(plt.imshow, cmap=plt.cm.Blues)
    s = plt.show


    mode = 'G'

    # make true if you want to restrict each filter to unextracted targets only 
    exclusivity = True

    scale_range = list(range(18,1,-1))

    theta_steps = 16
    length_ratio = 4

    cumulative = np.zeros((2200,2561), dtype='uint8')
    
    OUTPUT_DIR = 'scvme_output'
    SUBDIR = '_'.join((mode, '%02ds' % theta_steps,
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
    
    for sigma in scale_range:

        K1, K2 = load_curvatures(sigma, mode)
        
        targets = get_targets(K1,K2)
        
        savefile = ''.join(('%02d' % sigma, '-raw.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile), targets,
                cmap=plt.cm.Blues)

        if exclusivity:
            # remove any previously extracted vessels from consideration
            targets[cumulative.nonzero()] = False

        extracted = vessel_filter(targets, sigma, length_ratio, theta_steps)
        
        savefile = ''.join(('%02d' % sigma, '.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile), extracted,
            cmap=plt.cm.Blues)
        
        # label regions with this sigma in the cumulative map, but only where no label exists
        new_labels = sigma*np.logical_and(extracted != 0, cumulative == 0)
        cumulative += new_labels.astype('uint8')
    
    print('building labeled/cumulative map')

    c_img = label2rgb(cumulative, bg_label=0)
    
    bins = np.unique(c_img)
    
    # make a label set
    #cols = label2rgb(bins, bg_label=0)
    #cols = np.repeat(cols, 20, axis=0)
    #np.tile(np.expand_dims(cols, axis=1), (1,30,1))

    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative.png'), c_img)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative_binary.png'),
                            cumulative!=0, cmap=plt.cm.Blues)
