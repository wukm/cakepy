#!/usr/bin/env python3

import numpy.ma as ma
import numpy as np

from get_base import *
from curvemap import principal_curvatures

from skimage.exposure import rescale_intensity

import matplotlib.pyplot as plt

from skimage.morphology import remove_small_holes, remove_small_objects

from skimage.morphology import label

from functools import partial

bimshow = partial(plt.imshow, cmap=plt.cm.binary)

img = get_masked_raw()

try:
    I = np.load('claheraw.npy')
except FileNotFoundError:
    I = clahe_each(img)
    np.save('claheraw.npy', I)


I = ma.masked_array(I, mask=img.mask)

#try:
#    K = np.load('K_sigma_0100.pickle')
#except FileNotFoundError:
#    K1, K2 = principal_curvatures(I)
#    K = np.dstack((K1,K2))
#    K.dump('K_sigma_0100.pickle')
#else:
#    K1, K2 = np.dsplit(K)

K1, K2 = principal_curvatures(I, sigma=10)

R_B = (K1**2) / (K2**2)
S = K1**2 + K2**2

cond = (K2 < 0)

beta, c = 0.5**2, 15**2

V = np.exp(- R_B / (2*beta)) * (1 - np.exp(-S / (2*c))) * cond

k1, k2 = np.abs(K1), np.abs(K2)

#W = (cond * (k1 < ma.median(k1)) * (R_B < ma.median(R_B))).all(axis=-1)
W = (cond * R_B <  R_B.mean()).all(axis=-1)

L = label(W)

def showgrid(A):
    """
    show a large 2D binary masked matrix as 4 separate figures
    """
    
    try:
        a = A.filled(0)
    except AttributeError:
        a = A

    quads = [ _ for s in np.array_split(a,2,axis=1)
                for _ in np.array_split(s,2,axis=0)]

    for i, q in enumerate(quads):

        fig = plt.figure(i, dpi=165)
        plt.imshow(q, cmap=plt.cm.binary)
        fig.tight_layout()
        plt.show(block=False)


    


