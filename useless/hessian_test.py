#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi

from skimage import color, draw, exposure, feature, filters, graph, measure
from skimage import morphology, restoration, segmentation, transform, util

from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt

from skimage.morphology import remove_small_holes, remove_small_objects

from functools import partial

bimshow = partial(plt.imshow, cmap=plt.cm.binary)

def scrub(A, iterations=1):
    
    a = A.copy()

    for i in range(iterations*2):

        h = remove_small_holes(a, min_size=200, connectivity=2)
        b = remove_small_objects(a, min_size=200, connectivity=2)
        
        if not i % 2:
            a = h * b 
        else:
            a = h+b
            a[np.nonzero(a)] = 1

    return a

def get_bg_mask(img):
    
    bg_mask = img.any(axis=-1)
    bg_mask = 1 - bg_mask # consistent with np.ma, True if masked
    bg_mask = np.repeat(bg_mask[:,:,np.newaxis], 3, axis=2)  # make multichannel

    bound = segmentation.find_boundaries(bg_mask, mode='inner', background=1)
    bg_mask[bound] = 1
    
    holes = morphology.remove_small_holes(bg_mask)
    bg_mask[holes] = 1

    return bg_mask

# multichannel everything
img = ndi.imread('samples/raw.png', mode='RGB')

bg_mask = get_bg_mask(img)

img = util.img_as_float(img)

I = ma.masked_array(img, mask=bg_mask) 

##############################################

I_clahe = np.zeros_like(img, dtype='float')
for ic in range(3):
    channel = I[:,:,ic].filled(0)
    channel = exposure.equalize_adapthist(channel) #CLAHE
    I_clahe[:,:,ic] = channel

for i, sigma in enumerate([0.1, 0.5, 1, 1.2, 2, 3, 5, 8, 10]):

    H1 = np.zeros_like(img, dtype='float')
    H2 = np.zeros_like(img, dtype='float')
    signage = np.zeros_like(img, dtype='float')

    for ic in range(3):
        channel = I_clahe[:,:,ic]

        hes = feature.hessian_matrix(channel, sigma=sigma)

        # l1 is the larger and l2 is smaller, but this includes sign
        l1,l2 = feature.hessian_matrix_eigvals(*hes)

        L = np.dstack((l1,l2))

        # sort them by magnitude but keep signage in place
        mag = np.argsort(abs(L), axis=-1)
        neg = L < 0 

        # this is now a 2D mask that indicates where the largest eigenvalue by
        # magnitude has a negative sign. in the frangi filter, only these pixels are
        # considered to potentially be vascular
        l2_is_negative = (mag * neg).sum(axis=-1)
        
        
        # H2 is larger eigenvalue in absolute value, as consistent with Frangi paper
        H1[:,:,ic] = np.abs(L[:,:,0])
        H2[:,:,ic] = np.abs(L[:,:,1])
        signage[:,:,ic] = l2_is_negative


    H1 = ma.masked_array(H1, mask=I.mask)
    H2 = ma.masked_array(H2, mask=I.mask)

    # for display only
    #h1 = rescale_intensity(H1, in_range=(H1.min(), H1.max()), out_range=(0,1))
    #h2 = rescale_intensity(H2, in_range=(H2.min(), H2.max()), out_range=(0,1))

    Hd = H1 / H2
    A = (signage*(H1 < H1.mean())*(Hd < ma.median(Hd))).all(axis=-1)

    plt.figure(i)
    tmp = scrub(A)
    bimshow(tmp)
    plt.imsave('A_{}_noscrub.png'.format(sigma), A, cmap=plt.cm.binary)
#    tmp2 = scrub(tmp)
#    plt.imsave('A2_{}.png'.format(sigma), tmp2, cmap=plt.cm.binary)

    plt.show(block=False)
