#!/usr/bin/env python3

import matplotlib.pyplot as plt
from get_base import get_masked_raw

from skimage.data import camera
from functools import partial

from skimage.filters import gaussian
import scipy.fftpack as fftpack

from scipy.fftpack import fft2, ifft2, rfft, irfft, fftshift, ifftshift

from scipy import signal

from itertools import combinations_with_replacement

import numpy as np

g = partial(plt.imshow, cmap=plt.cm.gray)
s =  plt.show



def gauss_freq(shape, σ=1.):
    
    M,  N = shape
    fgauss = np.fromfunction(lambda μ,ν: ((μ+M+1)/2)**2 + ((ν+N+1)/2)**2, shape=shape)
    
    coeff = (1 / (2*np.pi * σ**2))
    return np.exp(-fgauss / (2*σ**2))

def blur(img, sigma):

    I = fftpack.fft2(img)
    #I = fftpack.rfft(img)
    
    # do whatever

    I *= gauss_freq(I.shape, sigma)
    

    return fftpack.ifft2(I).real

def fftgauss(img,sigma):

    """https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.signal.fftconvolve.html"""

    kernel = np.outer(signal.gaussian(img.shape[0], sigma),
                        signal.gaussian(img.shape[1],sigma))

    return signal.fftconvolve(img, kernel, mode='same')

def fft_hessian(image, sigma=1):
    """
    a reworking of skimage.feature.hessian_matrix that uses
    a FFT to compute gaussian
    """

    gaussian_filtered = fftgauss(image, sigma=sigma)
    
    Lx, Ly = np.gradient(gaussian_filtered)

    Lxx, Lxy = np.gradient(Lx)
    Lxy, Lyy = np.gradient(Ly)

    return (Lxx, Lxy, Lyy)
    
if __name__ == "__main__":
    
    C = camera() / 255
    P = get_masked_raw(mode='G') / 255
    p = P.filled(0)
    
    img = C
    
    outputs = (fftgauss(img, .01),
               fftgauss(img, 10),
               fftgauss(img, 30),
               fftgauss(img, 200)
               )
    
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(outputs[0], cmap='gray')
    axes[0, 0].set_title('standard σ=2')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(outputs[1], cmap='gray')
    axes[0, 1].set_title('standard σ=8')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(outputs[2], cmap='gray')
    axes[1, 0].set_title('FFT σ=2')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(outputs[3], cmap='gray')
    axes[1, 1].set_title('FFT σ=8')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

