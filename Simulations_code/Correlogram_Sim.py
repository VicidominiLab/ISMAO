# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:25:30 2024

@author: ffersini
"""

import numpy as np
import brighteyes_ism.simulation.PSF_sim as ism
import random 
import tensorflow as tf
import Simulation_Laucher as lc
import Find_Correlogram as cr
import SampleSimu_Laucher as lsg

#%%Corr2D_generator

def Corr_generator(amplitude, number, nz, var):
        
    exPar, emPar, grid_simul,grid_sample, noise = lc.Parameters(amplitude, number, nz)
    
    correlograms =  np.zeros((grid_sample.crp,grid_sample.crp,grid_simul.N*grid_simul.N, grid_simul.Nz))
        
    fp_f = np.zeros((grid_simul.N*grid_simul.N, grid_simul.Nz))
    
    img_f =  np.zeros((grid_sample.FOV, grid_sample.FOV, grid_simul.N*grid_simul.N, grid_simul.Nz))
    
    grid_sample.photon_flux = random.randrange(30, 50, 10)*1e6
    grid_sample.dwell_time  = random.randrange(50, 60, 10)*1e-6
    grid_simul.na  = np.round(random.randrange(12, 15, 1)*1e-1,2)
    exPar.na = grid_simul.na
    emPar.na = grid_simul.na

    ft_sample, depthmap = lsg.sample_generator(grid_sample, var)
    noise_count = np.multiply(noise,grid_sample.dwell_time)
    
    PSFs, detPSF, exPSF = ism.SPAD_PSF_3D(grid_simul , exPar, emPar, stedPar=None, spad=None,stack='symmetrical',normalize = True)    
    PSFs = np.reshape(PSFs,(grid_simul.Nz,grid_simul.Nx,grid_simul.Nx,grid_simul.N,grid_simul.N))
    PSFs = np.flip(PSFs, axis = -1)
    PSFs = np.flip(PSFs, axis = -2)
    PSFs = np.reshape(PSFs,(grid_simul.Nz, grid_simul.Nx, grid_simul.Nx, grid_simul.N*grid_simul.N))
    sz = PSFs.shape
        
    for i in range(0,grid_simul.Nz):
        PSF_padded= np.zeros((grid_simul.Nz, grid_sample.FOV, grid_sample.FOV, grid_simul.N*grid_simul.N)) 
        img = np.zeros((grid_sample.FOV, grid_sample.FOV, grid_simul.N*grid_simul.N)) 

        for ch in range(0,grid_simul.N*grid_simul.N):

            PSF_padded[i,:,:,ch] = np.pad(PSFs[i,:,:,ch],(((grid_sample.FOV - sz[1])//2), ((grid_sample.FOV - sz[1])//2)), mode = 'constant', constant_values=0)
            PSF_fft = tf.signal.rfft2d(PSF_padded[i,:,:,ch], fft_length=[2*grid_sample.FOV-1, 2*grid_sample.FOV-1])
            dummy = tf.signal.irfft2d(ft_sample*PSF_fft,fft_length= [2*grid_sample.FOV-1, 2*grid_sample.FOV-1])
            img[:,:,ch] = dummy[int(grid_sample.FOV/2):int(3*grid_sample.FOV/2), int(grid_sample.FOV/2):int(3*grid_sample.FOV/2)]        
        
        img[img<0]=0
        img = np.uint(np.round(img) )
        img = np.array(np.random.poisson(lam = img))
        noise_dset = np.random.poisson(lam = noise_count, size = [img.shape[0], img.shape[1], img.shape[2]])
        img = img + noise_dset
        crg_beads = cr.Find_Cor(img,grid_simul.pxsizex,exPar.na,eps = 1e-7)
        norm_cr = cr.crop_image(crg_beads,grid_sample.crp)           
        
        if grid_sample.fp_norm == True:
            fp = np.sum(img, axis=(0,1))
            fp = fp / np.max(fp)
            for ch in range(0,25):
                norm_cr[:,:,ch] *= (fp[ch])
            
            norm_cr -= np.min(norm_cr)
            norm_cr /= (np.max(norm_cr) - np.min(norm_cr))
        
        fp_f[:,i] = fp
        correlograms[:,:,:,i] = norm_cr
        img_f[:,:,:,i] = img
            
    print('Aberrations :', exPar.abe_ampli)
    print('Aberrations #:', exPar.abe_index)
    
    return PSFs, correlograms, fp_f, img_f
