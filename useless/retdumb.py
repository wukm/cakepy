#!/usr/bin/env python3

import scipy.ndimage as ndi
import skimage.feature
import skimage.filters
import skimage.exposure
import skimage.morphology
import numpy as np

from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt
import matplotlib
import skimage.draw

import numpy.ma as ma

from skimage.util import img_as_float

from skimage.segmentation import find_boundaries

img = ndi.imread('samples/raw.png', mode='RGB')

# get background mask depending on multichannel or not
# note that this is very simplistic and assumes the content does not have any
# black pixels in interior

# consistent with numpy.ma, True represented a **masked** element
if img.ndim == 3:
    bg_mask = img.any(axis=-1)
    bg_mask = 1 - bg_mask
    # make multichannel
    bg_mask = np.repeat(bg_mask[:,:,np.newaxis], 3, axis=2)
else:
    bg_mask = (img != 0)

img = img_as_float(img) # convert to [0,1] range

R = np.zeros_like(img)

fudge = 0.1 # prevent divide by zero?

for sigma in [0.05, 0.1, 0.15, 0.5, 1]:
    C = skimage.filters.gaussian(img + fudge, sigma, mode='constant', multichannel=True)
    C /= (2*np.pi * sigma**2) # normalization constant
    R += np.log(img+fudge) - np.log(C)

img = ma.masked_array(img, mask=bg_mask)

R = ma.masked_array(R, mask=bg_mask)

bound = find_boundaries(R.mask, mode='thick')
R[bound] = ma.masked

r = rescale_intensity(R, in_range=(R.min(), R.max()))
#closure = R <= R.mean() # this is all BG (non vessels)
#
## maybe this can be repeated multiple times for better results
#closure = skimage.morphology.binary_closing(closure)
#
#closure = bg_mask*(1-closure) # this is a dull approx of the arteries
#
#
#fig, axes = plt.subplots(1,2,sharex=True,sharey=True,
#                         subplot_kw={'adjustable':'box-forced'})
#
#axes[0].imshow(closure, cmap=plt.cm.gray)
#axes[0].set_title("original (L)")
#axes[1].imshow(closure * img, cmap=plt.cm.gray)
#
#axes[1].set_title("retdumb")
#
#fig.tight_layout()
#fig.show()
#
#fig, axes = plt.subplots(1,2,sharex=True,sharey=True,
#                         subplot_kw={'adjustable':'box-forced'})
#
#dimg = skimage.exposure.rescale_intensity((img/255)**3)
#axes[0].imshow(dimg, cmap=plt.cm.gray)
#axes[0].set_title("squared(L)")
#axes[1].imshow(closure * dimg,cmap=plt.cm.gray)
#
#axes[1].set_title("retdumb")
#
#fig.tight_layout()
#fig.show()
