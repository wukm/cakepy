#!/usr/bin/env python3

from get_base import get_trace
import numpy as np

def confusion(a, b=None, a_color=None, b_color=None):
    """
    visual confusion matrix for 2D boolean arrays of the
    same size
    """
    
    if a_color is None:
        a_color = np.array([0.9, 0.6, 0.1])
    if b_color is None:
        b_color = 1 - a_color
    
    if b is None:
        b = get_trace()

    a_c = np.tile(a[:,:,np.newaxis], (1,1,3)) * a_color
    b_c = np.tile(b[:,:,np.newaxis], (1,1,3)) * b_color

    return a_c + b_c

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    B = get_trace()
    A = np.random.random_integers(0,1,B.shape)*B

    C = confusion(A)

