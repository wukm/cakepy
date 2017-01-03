#!/usr/bin/env python3


import numpy as np
import numpy.ma as ma

from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from numpy.linalg import eig
from numpy.fft import fft2, ifft2, rfft2, irfft2
from numpy.fft import hfft, ihfft, fftshift, ifftshift
from functools import partial

def fast_hessian(img, sigma=3.0):
    """Calculates Hessian in *frequency space* via fft2

    -4π [   μ^2 iL(μ,ν) μν iL(μ,ν)  ]
        [   μν iL(μ,ν)  ν^2 iL(μ,ν) ] 

    where iL(μ,ν) is the inverse 2D FFT of the image

    """
    #ffx = partial(rfft2, s=img.shape)
    #iffx = partial(irfft2, s=img.shape)
    
    ffx = fft2
    iffx = ifft2
    iL = ffx(img)
    
    SHIFT = False

    # coordinate matrices [[ u^2, uv], [uv, v^2]]
    cxx = np.fromfunction(lambda i,j: i**2, iL.shape, dtype='float')
    cxy = np.fromfunction(lambda i,j: i*j, iL.shape, dtype='float')
    cyy = np.fromfunction(lambda i,j: j**2, iL.shape, dtype='float')
    
    if SHIFT:
        cxx = fftshift(cxx)
        cxy = fftshift(cxy)
        cyy = fftshift(cyy)

    # elementwise multiplication
    hxx = -4*np.pi**2 * cxx * iL
    hxy = -4*np.pi**2 * cxy * iL
    hyy = -4*np.pi**2 * cyy * iL
    
    exparg = -(cxx + cyy) / (2*np.pi**2 * sigma**2)
    #A = 1 / (2*np.pi*sigma**2)
    A = 1
    fgauss = A*np.exp(exparg)

    hxx = fgauss * hxx
    hxy = fgauss * hxy
    hyy = fgauss * hyy
    
    Hxx = iffx(hxx)
    Hxy = iffx(hxy)
    Hyy = iffx(hyy)

    if SHIFT:
        Hxx = ifftshift(Hxx)
        Hxy = ifftshift(Hxy)
        Hyy = ifftshift(Hyy)

    return  Hxx, Hxy, Hyy

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
        
        signed: (bool) Whether to return signed principal curvatures or simply
                magnitudes.

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
    """2D only, handles masked arrays""" 
    if H is None:
        Hxx, Hxy, Hyy = hessian_matrix(img, sigma)
    
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
        leading_thetas = ma.masked_array(leading_thetas, img.mask)
        trailing_thetas = ma.masked_array(trailing_thetas, img.mask)


    return trailing_thetas, leading_thetas

    

    
if __name__ == "__main__":

    
    from get_base import get_preprocessed
    import matplotlib.pyplot as plt
    from functools import partial
    from fpd import get_targets
    b = partial(plt.imshow, cmap=plt.cm.Blues) 
    sp = partial(plt.imshow, cmap=plt.cm.spectral) 
    s = plt.show

    img = get_preprocessed(mode='G')
    
    sigma = 1.
    print('σ=',sigma)
    print('calculating hessian H')
    H = hessian_matrix(img, sigma=sigma)
    print('calculating hessian via FFT (F)')
    F = fast_hessian(img, sigma=sigma)
    
    print('calculating principal curvatures for σ=1')
    K1,K2 = principal_curvatures(img, sigma=sigma, H=H)
    print('calculating principal curvatures for σ=1 (fast)')
    k1,k2 = principal_curvatures(img, sigma=sigma, H=F)
    
    plt.figure(0)
    sp(np.log(np.abs(K1)))
    plt.figure(1)
    sp(np.log(np.abs(k1)))
    #print('getting targets')
    #T = get_targets(K1,K2, threshold=False) 
    #t = get_targets(k1,k2, threshold=False) 

    #plt.figure(0)
    #b(T > T.mean())
    #plt.figure(1)
    #b(t > tfmean())
