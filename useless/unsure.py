#!/usr/bin/env python3

import scipy.ndimage as ndi
import skimage.feature
import skimage.exposure
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import skimage.draw

img = ndi.imread('samples/dogmo.png', mode='L')

#try:
#    A = np.load('dohblob_no_overlap.npy')
#except FileNotFoundError:
#    print('file not found, redoing')
#    A = skimage.feature.blob_doh(img, threshold=0.2, overlap=1.0)
#    np.save('dohblob_no_overlap.npy', A)
#else:
#    print('loaded file successfully')

#img = skimage.exposure.equalize_hist(img)

A = skimage.feature.blob_log(img, num_sigma=20, log_scale=True, overlap=1.0)
blobplot = np.zeros_like(img)

for row in A:
    row[2] *= np.sqrt(2) # radius of blob is roughly sqrt(2)*sigma
    r, c, radius = row.astype(np.int)
    B = skimage.draw.circle(r, c, radius, shape=blobplot.shape)
    blobplot[B] = 1
    blobplot[r,c] = 1

fig, axes = plt.subplots(1,2,sharex=True,sharey=True,
                         subplot_kw={'adjustable':'box-forced'})

axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].set_title("original (L)")
axes[1].imshow(blobplot, cmap=plt.cm.gray)
axes[1].set_title("LoG Blob")

fig.tight_layout()

