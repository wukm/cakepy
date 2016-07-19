#!/usr/bin/env python3

import scipy.ndimage as ndi
from frangi import *

from skimage import feature, exposure, draw, filters, morphology, segmentation, color
import skimage.exposure
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import skimage.draw

import skimage.filters
from skimage.exposure import rescale_intensity


img = ndi.imread('samples/raw.png', mode='RGB')

I = np.zeros_like(img)
W = np.zeros_like(img)

for ic in range(3):
    gimg = img[:,:,ic]
    I[:,:,ic] = filters.rank.autolevel(gimg, morphology.disk(15))
    W[:,:,ic] = morphology.white_tophat(I[:,:,ic],selem=morphology.disk(5)) 


B = color.rgb2gray(I-W)
Hxx, Hxy, Hyy = feature.hessian_matrix(B)
L1, L2 = feature.hessian_matrix_eigvals(Hxx,Hxy,Hyy)


fig, ax = plt.subplots(2, 2, figsize=(5, 4), sharex=True, sharey=True,
                       subplot_kw={'adjustable': 'box-forced'})
ax = ax.ravel()

ax[0].imshow(rescale_intensity(abs(L1)))
ax[1].imshow(rescale_intensity(abs(L2)))
ax[2].imshow(rescale_intensity(np.sqrt(L1**2 + L2**2)))
ax[3].imshow(rescale_intensity(np.abs(L1) / (np.abs(L2)+0.1)))

edges = rescale_intensity(abs(L1))
dogmo = ndi.imread('retdumb.png', mode='L')

markers = np.zeros_like(dogmo)

markers[dogmo < 30] = 2 #background
markers[dogmo > dogmo.mean()] = 1 #foreground

ws = morphology.watershed(edges, markers)
seg1 = ndi.label(ws == 1)[0]

stuff = color.label2rgb(seg1, image=dogmo, bg_label=0)

#F = frangi(gimg, scale_range=(4,6), scale_step=2, black_ridges=False)

#p2, p98 = np.percentile(F, (2,98))

#FF = rescale_intensity(F, in_range=(p2,p98))

#plt.imshow(FF, cmap=plt.cm.gray)
