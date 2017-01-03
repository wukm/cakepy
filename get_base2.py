#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi

from skimage import color, draw, exposure, feature, filters, graph, measure
from skimage import morphology, restoration, segmentation, transform, util

from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.color.adapt_rgb import adapt_rgb, each_channel
import matplotlib.pyplot as plt

def get_bg_mask(img):
    
    #if img.ndim == 3:
    #    bg_mask = img.any(axis=-1)
    #    bg_mask = np.invert(bg_mask) # consistent with np.ma, True if masked

    #    # make multichannel (is it really this hard?)
    #    bg_mask = np.repeat(bg_mask[:,:,np.newaxis], 3, axis=2) 
    #
    #else:
    #    bg_mask = (img != 0)
    #    bg_mask = np.invert(bg_mask) # see above

    #bound = segmentation.find_boundaries(bg_mask, mode='inner', background=1)
    #bg_mask[bound] = 1
    #min_size = img.shape[0] * img.shape[1] // 4 
    #holes = morphology.remove_small_holes(bg_mask, min_size=min_size)
    #bg_mask[holes] = 1
    
    bg_mask = segmentation.find_boundaries(img)
    bg_mask = morphology.remove_small_objects(bg_mask)
    bg_mask = morphology.remove_small_holes(bg_mask)

    bg_mask = np.invert(bg_mask)
    return bg_mask


def get_trace(tracefile=None):
    
    if tracefile is None:
        tracefile = 'samples/trace.png'
    t = ndi.imread(tracefile, 'L');
    t = t.astype('bool')

    return np.invert(t)

def get_masked_raw(filename=None, mode='RGB'):
    
    if mode == 'G':
        readmode = 'RGB'
    else:
        readmode = mode
    
    if filename is None:
        filename = 'samples/raw.png'

    img = ndi.imread(filename, mode=readmode)
    
    if mode == 'G':
        img = img[:,:,1] # green channel only (disregard rest)

    bg_mask = get_bg_mask(img)

    I = ma.masked_array(img, mask=bg_mask) 

    return I

 
@adapt_rgb(each_channel)
def clahe_each(img):
    
    # note you need to fix the fucking mask after this :/
    return exposure.equalize_adapthist(img)

def preprocess(img):
    
    if img.ndim == 3:
        I = clahe_each(img)
    else:
        I = equalize_adapthist(img)
    
    try:
        mask = img.mask
    except AttributeError:
        pass
    else:
        I = ma.masked_array(I, mask=mask)

    return I

def get_preprocessed(filename=None, mode='L'):
    

    img = get_masked_raw(filename=filename, mode=mode)
    I = preprocess(img)
    return I

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    filename = 'samples/nb.jpg'
    img = get_masked_raw(filename, mode='G')
    I = get_preprocessed(filename, mode='G')

