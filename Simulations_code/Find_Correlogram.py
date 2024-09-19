# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:39:46 2024

@author: ffersini
"""

import numpy as np
from scipy.ndimage import fourier_shift
from skimage.filters import gaussian
           
def sigmoid(R, T, S):
    return  1 / (1 + np.exp( (R-T)/S ) )

def Low_pass(img, T, S, data = 'real'):
    
    if data == 'real':
        img_fft = np.fft.fftn(img, axes = (0,1) )
        img_fft = np.fft.fftshift(img_fft, axes = (0,1) )
    elif data == 'fourier':
        img_fft = img
    else:
        raise ValueError('data has to be \'real\' or \'fourier\'')
            
    Nx = np.shape(img_fft)[0]
    Ny = np.shape(img_fft)[1]
    cx = int ( ( Nx + np.mod(Nx,2) ) / 2)
    cy = int ( ( Ny + np.mod(Ny,2) ) / 2)
    
    x = ( np.arange(Nx) - cx ) / Nx
    y = ( np.arange(Ny) - cy ) / Ny
    
    X, Y = np.meshgrid(x, y)
    R = np.sqrt( X**2 + Y**2 )
    
    sig = sigmoid(R, T, S)
    
    img_filt = np.einsum( 'ij..., ij -> ij...', img_fft, sig )
    
    if data == 'real':
        img_filt = np.fft.ifftshift(img_filt, axes = (0,1) )
        img_filt = np.fft.ifftn(img_filt, axes = (0,1) )
        img_filt = np.abs(img_filt)
    
    return img_filt, sig

def hann2d( shape ):

    Nx, Ny = shape[0], shape[1]
    

    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x,y)
    W = 0.5 * ( 1 - np.cos( (2*np.pi*X)/(Nx-1) ) )
    W *= 0.5 * ( 1 - np.cos( (2*np.pi*Y)/(Ny-1) ) )
    
    return W

def rotate(array, degree):
    radians = degree*(np.pi/180)  
    x = array[:,0]
    y = array[:,1]    
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, ( [x, y] ) )
    return m


#%% Data and settings

def Find_Cor(Input_img, pxsize, NA = 1.4, lowpass = True, norm = True, eps = 1e-7, ref = 12 )    :

    '''Gaussian Filter''' 
    # Input_img = gaussian(Input_img, sigma = 2, channel_axis = -1)

    wavelength = 515 # nm
    t = 2*(2*NA/wavelength) * pxsize
    s = 0.01
    ref = 12   
    
    '''Apodization'''    
    W  = hann2d(np.shape(Input_img))
    img = np.einsum( 'ijk,ij -> ijk', Input_img, W)
      
    '''Fourier Tansform'''     
    img_fourier = np.fft.fftn(img, axes = [0,1])
    img_fourier = np.fft.fftshift(img_fourier, axes = [0,1])
    # img_fourier_t = img_fourier
    '''Low-pass filter input images'''  
    if lowpass == True:
        for i in range(0,25): 
            img_fourier[:,:,i], sig = Low_pass(img_fourier[:,:,i], t, s, data = 'fourier')
            
    '''Correlogram calculation'''
    
    img_correlation =  np.einsum( 'ijk,ij -> ijk', img_fourier, np.conj(img_fourier[:,:,ref]))
    
    if norm == True:
        correlogram = np.where(np.abs(img_correlation) < eps, 0, img_correlation / np.abs(img_correlation))
    else:
        correlogram = img_correlation
    # correlogram = img_correlation / np.maximum( np.abs(img_correlation), eps)
        
    '''Low-pass filter input images'''  
    if lowpass == True:
        for i in range(0,25): # low-pass filter correlograms before ifft
            correlogram[:,:,i], sig = Low_pass(correlogram[:,:,i], t, s, data = 'fourier')

    if norm == True:
        sz = correlogram.shape
        
        crg_padded = np.zeros( (sz[0]*3, sz[1]*3, sz[2]), dtype = np.complex128)
        
        for n in range(sz[-1]):
            crg_padded[:,:,n] = np.pad( correlogram[:,:,n], (sz[0], sz[1]), mode = 'constant', constant_values = 0)
            
        # crg_padded = da.padding(correlogram, pad = sz[0])
        
    else:
        
        crg_padded = correlogram
    
    '''Fourier Anti-Tansform'''
    
    crg_padded = np.fft.ifftn(crg_padded, axes = [0,1])
    crg_padded = np.abs( np.fft.ifftshift(crg_padded, axes = [0,1]) )
        
    return crg_padded
    # return crg_padded,W,img_fourier_t,img_fourier

    
#%%

def crop_image(dset,n):
    
    width, height, channel = dset.shape
    width = width//2
    height = height//2
    num_pix = n//2
    croped_df = dset[width-num_pix:width+num_pix,height-num_pix:height+num_pix,:]
    
    return croped_df