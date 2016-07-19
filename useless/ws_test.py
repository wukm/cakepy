#!/usr/bin/env python3

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np

from skimage import util

from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

from skimage.morphology import watershed, disk
from skimage.filters import rank

from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from skimage.segmentation import random_walker

# convenience function
gray = plt.cm.gray

img_raw = ndi.imread('raw.png', mode='RGB')


bg_mask = img_raw.any(axis=2) # completely black pixels (hope it's on the edge)

img = img_raw[:,:,1] # green channel only to use for 2D processes (for now)

# note that rank.median wants uint8 or uint16
    
denoised = rank.median(img, disk(2))

grad = rank.gradient(denoised, disk(5))

#plt.imshow(grad < 10)

H = hessian_matrix(img) # see optional args

l1, l2 = hessian_matrix_eigvals(*H)

L = np.abs(l1)

# use a different method-- this is for uint8 only
LG = rank.gradient(L, disk(3))

# ridges of some sort
T = LG < LG.mean()



#markers, _ = ndi.label(T)

#ws = watershed(LG, markers, mask=bg_mask)

markers = np.zeros_like(img)
markers[L < L.mean()*.5] = 1
markers[L > L.mean()] = 2
markers[1-bg_mask] = -1
