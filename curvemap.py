#!/usr/bin/env python3


import numpy as np
import numpy.ma as ma

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.linalg import eig
from functools import partial

from hfft import fft_hessian


def principal_curvatures(img, sigma=1.0, H=None):
    """
    Return the principal curvatures {κ1, κ2} of an image, that is, the
    eigenvalues of the Hessian at each point (x,y). The output is arranged such
    that |κ1| <= |κ2|.

    Input:

        img:    An ndarray representing a 2D or multichannel image. If the image
                is multichannel (e.g. RGB), then each channel will be proccessed
                individually. Additionally, the input image may be a masked
                array-- in which case the output will preserve this mask
                identically.
                
                PLEASE ADD SOME INFO HERE ABOUT WHAT SORT OF DTYPES ARE
                EXPECTED/REQUIRED, IF ANY

        sigma:  (optional) The scale at which the Hessian is calculated.
        
        H:      (optional) provide sigma (else it will be calculated)
    Output:
        
        (K1, K2):   A tuple where K1, K2 each are the exact dimension of the
                    input image, ordered in magnitude such that |κ1| <= |κ2|
                    in all locations. If *signed* option is used, then elements
                    of K1, K2 may be negative.
    
    Example:
        
        >>> K1, K2 = principal_curvatures(img)
        >>> K1.shape == img.shape
        True
        >>> (K1 <= K2).all()
        True

        >> K1.mask == img.mask
        True
    """

    # determine if multichannel
    multichannel = (img.ndim == 3)
    
    if not multichannel:
        # add a trivial dimension
        img = img[:,:,np.newaxis]

    K1 = np.zeros_like(img, dtype='float64')
    K2 = np.zeros_like(img, dtype='float64')

    for ic in range(img.shape[2]):

        channel = img[:,:,ic]

        # returns the tuple (Hxx, Hxy, Hyy)
        if H is None:
            H = hessian_matrix(channel, sigma=sigma)
        
        # returns tuple (l1,l2) where l1 >= l2 but this *includes sign*
        L = hessian_matrix_eigvals(*H)
        L = np.dstack(L)

        mag = np.argsort(abs(L), axis=-1)
        
        # just some slice nonsense
        ix = np.ogrid[0:L.shape[0], 0:L.shape[1], 0:L.shape[2]]
        
        L = L[ix[0], ix[1], mag]

        # now k2 is larger in absolute value, as consistent with Frangi paper

        K1[:,:,ic] = L[:,:,0]
        K2[:,:,ic] = L[:,:,1]

    try:
        mask = img.mask
    except AttributeError:
        pass
    else:
        K1 = ma.masked_array(K1, mask=mask)
        K2 = ma.masked_array(K2, mask=mask)
    
    # now undo the trivial dimension
    if not multichannel:
        K1 = np.squeeze(K1)
        K2 = np.squeeze(K2)

    return K1, K2

def principal_directions(img, sigma, H=None):
    """
    will ignore calculation of principal directions of masked areas

    despite the name, this function actually returns the theta corresponding to
    leading and trailing principal directions, i.e. angle w / x axis
    """

    if H is None:
        H = hessian_matrix(img, sigma)

    Hxx, Hxy, Hyy = H
    

    # check if input image is masked
    try:
        mask = img.mask
    except AttributeError:
        masked = False
    else:
        masked = True

    dims = img.shape

    # where to store
    trailing_thetas = np.zeros_like(img)
    leading_thetas = np.zeros_like(img)


    # maybe implement a small angle correction
    for i, (xx, xy, yy) in enumerate(np.nditer([Hxx, Hxy, Hyy])):
        
        # grab the (x,y) coordinate of the hxx, hxy, hyy you're using
        subs = np.unravel_index(i, dims)
        
        # ignore masked areas (if masked array)
        if masked and img.mask[subs]:
            continue

        h = np.array([[xx, xy], [xy, yy]]) # per-pixel hessian
        l, v = eig(h) # eigenvectors as columns
        
        # reorder eigenvectors by (increasing) magnitude of eigenvalues
        v = v[:,np.argsort(np.abs(l))]
        
        # angle between each eigenvector and positive x-axis
        # arccos of first element (dot product with (1,0) and eigvec is already
        # normalized)
        trailing_thetas[subs] = np.arccos(v[0,0]) # first component of each
        leading_thetas[subs] = np.arccos(v[0,1]) # first component of each
    
    if masked:
        leading_thetas = ma.masked_array(leading_thetas, mask)
        trailing_thetas = ma.masked_array(trailing_thetas, mask)


    return trailing_thetas, leading_thetas

    

    
if __name__ == "__main__":

    
    from get_base import get_preprocessed
    import matplotlib.pyplot as plt
    from functools import partial
    from fpd import get_targets
    b = partial(plt.imshow, cmap=plt.cm.Blues) 
    sp = partial(plt.imshow, cmap=plt.cm.spectral) 
    s = plt.show
    
    import time 

    img = get_preprocessed(mode='G')
    
    for sigma in [0.5, 1, 2, 3, 5, 10]:

        print('-'*80)
        print('σ=',sigma)
        print('calculating hessian H')

        tic = time.time()
        H = hessian_matrix(img, sigma=sigma)

        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()
        print('calculating hessian via FFT (F)')
        h = fft_hessian(img, sigma)
        
        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()
        print('calculating principal curvatures for σ={}'.format(sigma))
        K1,K2 = principal_curvatures(img, sigma=sigma, H=H)
        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()
        print('calculating principal curvatures for σ={} (fast)'.format(sigma))
        k1,k2 = principal_curvatures(img, sigma=sigma, H=h)

        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()
        
        #####

        print('calculating targets for σ={}'.format(sigma))
        T = get_targets(K1,K2, threshold=False) 

        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()

        print('calculating targets for σ={} (fast)'.format(sigma))
        t = get_targets(k1,k2, threshold=False) 

        toc = time.time()
        print('time elapsed: ', toc - tic)
        
        ###### 
        
        print('extending masks')
        
        # extend mask over nontargets items
        img1 = ma.masked_where( T < T.mean(), img)
        img2 = ma.masked_where( t < t.mean(), img)

        tic = time.time()
        print('calculating principal directions for σ={}'.format(sigma))
        T1,T2 = principal_directions(img1, sigma=sigma, H=H)
        toc = time.time()
        print('time elapsed: ', toc - tic)
        tic = time.time()

        print('calculating principal directions for σ={} (fast)'.format(sigma))
        t1,t2 = principal_directions(img2, sigma=sigma, H=h)
        toc = time.time()
        print('time elapsed: ', toc - tic)
