#!/usr/bin/env python3


import numpy as np
import numpy.ma as ma

from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def principal_curvatures(img, sigma=1.0):
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
    
if __name__ == "__main__":

    
    from get_base import get_preprocessed
    I = get_preprocessed(mode='L')

    K1, K2 = principal_curvatures(I)    
    
    R_B = np.abs(K1 / K2).filled(0)
    S = np.sqrt(K1**2 + K2**2).filled(0)

