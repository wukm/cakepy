#!/usr/bin/env python3

"""
This is a multiscale Frangi filter modified to use at-scale morphological
arguments based on principal direction
"""

import os
import os.path
import datetime

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


def load_principal_data(sigma, subdir=None, mode='G'):
    """ 
    load principal curvature and principal direction matrices for a given scale
    space from an existing pickle. Does not do any error handling currently, so
    please make sure that the files exist first and be aware of the filename
    formats used (this behavior should be fixed in the future).
    
    INPUT:

    sigma:  the scale space
    mode:   one of 'G', 'L', or 'RGB'
    
    OUTPUT:

    K1, K2, T, L
        K1, K2: principal curvatures at each pixel, arranged such that
                (np.abs(K1) <= np.abs(K2)).all() -> True
        T:      the "trailing" principal direction (corresponding to K1)
                at each pixel, given a θ in [0, π) measuring the angle
                between the eigenvector and the positive x-axis.
        L:      the "leading" principal direction (corresponding to K2),
                as above.

        Note that each principal direction is only given between [0,π) such
        that the direction (but not orientation) is conserved. T,L should
        be orthogonal to each other at each pixel.
    """
    
    if subdir is None:
        subdir = ''

    curve_datadir = os.path.join('clahe_curvatures', subdir)
    theta_datadir = os.path.join('clahe_directions', subdir)

    
        
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
    mask = img.mask
    extracted = np.zeros_like(img)
    img = binary_erosion(img, selem=disk(sigma))
    #img = remove_small_objects(img, min_size=sigma**3)
    img = binary_dilation(img, selem=disk(sigma))


    width, length = int(2*sigma), int(sigma*length_ratio)

    if length == 0:
        length = 1

    rect = rectangle(width, length)
    outer_rect =  rectangle(width+2*sigma+4,length)
    outer_rect[sigma:-sigma,:] = 0
    thetas = np.round(thetas*180 / np.pi)

    # this should behave the same as thetas[thetas==180] = 0
    # but not return a warning
    thetas.put(thetas==180, 0) # these angles are redundant

    if verbose:
        print('running vessel_filter with σ={}: w={}, l={}'.format(
            sigma, width,length), flush=True)

    if verbose:
        print('building rotated filters...', end=' ')

    srot = partial(rotate, resize=True, preserve_range=True) # look at order
    rotated = [srot(rect, theta) for theta in range(180)]
    
    if verbose:
        print('done.')
    
    if verbose:
        print('building outer filters...', end=' ')
    outer_rotated = [srot(outer_rect, theta) for theta in range(180)]

    for theta in range(180):
        if verbose:
            print('θ=', theta, end='\t', flush=True)
            if theta % 6 == 0:
                print()

        vessels = binary_erosion(img, selem=rotated[theta])
        #margins = binary_dilation(img, selem=outer_rotated[theta])
        #margins = np.invert(margins)
        #vessels = np.logical_and(vessels, margins)
        extracted = np.logical_or(extracted, (thetas == theta) * vessels) 
    if verbose:
        print('') # new line
    
    extracted = binary_dilation(extracted, selem=disk(sigma))
    extracted[mask] = 0
    return extracted

def get_targets(K1,K2, method='F'):
    """
    returns a binary threshold (conservative)

    F -> frangi filter with default arguments
    R -> blobness measure
    S -> anisotropy measure (greater than median)
    """ 
    if method == 'R':
        R = (K1 / K2) ** 2
        T = R < ma.median(R)
    elif method == 'S': 
        S = (K1**2 + K2**2)/2
        T = S > ma.median(S)
    elif method == 'F':
        R = (K1 / K2) ** 2
        S = (K1**2 + K2**2)/2
        beta, c = 0.5, 15
        F = np.exp(-R / (2*beta**2))
        F *= 1 - np.exp(-S / (2*c**2))
        T = (K2 < 0)*F
        T = T > (T[T != 0]).mean()
    else:
        raise('Need to select method as "F", "S", or "R"')
    
    return T

if __name__ == "__main__":
    
    from score import confusion

    b = partial(plt.imshow, cmap=plt.cm.Blues)
    s = plt.show


    imagename = 'TA-BN0657337'
    mode = 'G'
    method = 'F'

    # make true if you want to restrict each filter to unextracted targets only 
    exclusivity = False

    scale_range = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    length_ratio = .5
    from get_base import get_preprocessed
    base = get_preprocessed('samples/traces/TA-BN0657337.png', mode='G')

    cumulative = np.zeros(base.shape, dtype='uint8')
    extracted_all = np.zeros((base.shape[0], base.shape[1], len(scale_range)), dtype='bool') 
    OUTPUT_DIR = 'fpd_output'
    SUBDIR = '_'.join((imagename, mode, method))

    if exclusivity:
        SUBDIR = '_'.join((SUBDIR, 'exclusive'))
    
    n = datetime.datetime.now()

    SUBDIR = ''.join((SUBDIR, n.strftime('%y%m%d_%H%M')))

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

        K1, K2, L, T = load_principal_data(sigma, subdir=imagename, mode=mode)
        
        targets = get_targets(K1,K2,method='F')
        
        savefile = ''.join(('%02d' % sigma, '-raw.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile),
                targets*L, cmap=plt.cm.spectral)

        if exclusivity:
            # remove any previously extracted vessels from consideration
            targets[cumulative.nonzero()] = False

        extracted = vessel_filter(targets, L, sigma, length_ratio)
        extracted_all[:,:,n] = extracted
        savefile = ''.join(('%02d' % sigma, '.png'))
        plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, savefile),
                extracted, cmap=plt.cm.Blues)
        
        # label regions with this sigma in the cumulative map, but only where no label exists
        new_labels = sigma*np.logical_and(extracted != 0, cumulative == 0)
        cumulative += new_labels.astype('uint8')
          

    print('building labeled/cumulative map')

    c_img = label2rgb(cumulative, bg_label=0)
    
    bins = np.unique(c_img)
        
    #    # make a label set
    #    #cols = label2rgb(bins, bg_label=0)
    #    #cols = np.repeat(cols, 20, axis=0)
    #    #np.tile(np.expand_dims(cols, axis=1), (1,30,1))

    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative.png'), cumulative)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'cumulative_binary.png'),
                                cumulative!=0, cmap=plt.cm.Blues)

    C = confusion(cumulative!=0, b=base)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'confusion.png'), C)

    print('calculating basic skeleton and reduced skeleton') 

    full_skel = skeletonize(cumulative!=0)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'full_skel.png'),
                                    full_skel, cmap=plt.cm.Blues)
    skel = remove_small_objects(full_skel, min_size=50,
                            connectivity=2)
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'small_skel.png'),
                                    skel, cmap=plt.cm.Blues)
    
    print('trimming cumulative map to the reduced skeleton', end=' ', flush=True)
    matched_all = np.zeros_like(skel)

    for i, scale in enumerate(scale_range):

        print('σ', end=' ', flush=True)
        e = extracted_all[:,:,i]
        el, nl = label(e, return_num=True)
        matched = np.zeros_like(matched_all)

        for region in range(1, nl+1):
            if np.logical_and(el==region, skel).any():
                matched = np.logical_or(matched, el==region)
        
        matched_all = np.logical_or(matched_all, matched)

    print('\nsaving it all!')
    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'on_skel.png'),
                                    matched_all, cmap=plt.cm.Blues)

    #   plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'on_skel_confusion.png'),
    #                                confusion(matched_all))
    
    #c_img2 = label2rgb(cumulative*matched_all, bg_label=0)

    plt.imsave(os.path.join(OUTPUT_DIR, SUBDIR, 'on_skel_labeled.png'),
            cumulative*matched_all)

    # remainder stuff
    remainder_sigma = 0.1
    k1, k2, th1, th2 = load_principal_data(sigma=remainder_sigma, subdir='nb',
            mode=mode)

    rt = get_targets(k1,k2)

