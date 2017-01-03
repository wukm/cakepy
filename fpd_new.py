#!/usr/bin/env python3

from get_base import get_preprocessed
import matplotlib.pyplot as plt

from functools import partial
from fpd import get_targets, vessel_filter


import numpy as np
import numpy.ma as ma

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.linalg import eig

from hfft import fft_hessian
from curvemap import principal_curvatures, principal_directions


from score import confusion

b = partial(plt.imshow, cmap=plt.cm.Blues)
s = plt.show


mode = 'G' # use whatever channel
method = 'F' # method of hessian filter


# make true if you want to restrict each filter to unextracted targets only 
exclusivity = False


scale_range = np.arange(14,0,-1)

length_ratio = 0.5
img = get_preprocessed(mode='G')


h = fft_hessian(img,sigma)


k1,k2 = principal_curvatures(img, sigma=sigma,H=h)

t1,t2 = principal_directions(img,sigma=sigma,H=h)
t = get_targets(k1,k2, threshold=False)

# extend mask
timg = ma.masked_where(t < t.mean(), img)
t1,t2 = principal_directions(timg, sigma=sigma, H=h)

