#!/usr/bin/env python3

from glob import glob
import pickle
import os.path
import numpy as np
import numpy.ma as ma

VPD_DIR = 'vpd-G'
THETA_DIR = 'clahe_directions'

globs = sorted(glob(os.path.join(VPD_DIR, '*.pickle')))


L = np.zeros((2200, 2561))
for n, pick in enumerate(globs, 1):

    if n > 5:
        break

    with open(pick, 'rb') as f:

        p = pickle.load(f)

    L_new = p['L_var']
    L = np.dstack((L, L_new))

# this is stupid now
L = L[:,:,1:]
