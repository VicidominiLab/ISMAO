# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:45:15 2024

@author: ffersini
"""

import numpy as np
import brighteyes_ism.dataio.mcs as mcs
from ml_ism.PSF_estimator_lib import GridFinder
import brighteyes_ism.simulation.PSF_sim as psf
import Extra_Paramters as ts
import Download_data as dd
import os

#Set simulation paramters

def Parameters(amplitude, number, nz):
   
    '''Read Experimental dataset to simulate noise and collect setup info'''

    url_data = 'https://zenodo.org/records/13789465/files/noise_map.npy'
    name_data = 'noise_map.npy'
    if not os.path.isfile(name_data):
        dd.download(url_data, name_data)
    else:
        print('NoiseMap already downloaded.')

    noise = np.load(name_data).reshape(25)

    '''Read Experimental dataset to collect setup info'''

    url_data = 'https://zenodo.org/records/13789465/files/PSF.h5'
    name_data = 'PSF.h5'
    if not os.path.isfile(name_data):
        dd.download(url_data, name_data)
    else:
        print('PSF already downloaded.')

    dset, meta = mcs.load(name_data)
    dset = np.squeeze((dset.sum(axis = -2)))


    '''Excitation params'''
    exPar = psf.simSettings()
    exPar.na = 1.4   # numerical aperture
    exPar.wl = 488   # exc wavelength [nm]
    exPar.n = 1.5
    exPar.mask_sampl = 180
    exPar.mask = 'Zernike'
    exPar.w0 = 2.08 #(4.9/(np.sqrt(2*np.log(2))))/2
    exPar.abe_ampli = amplitude 
    exPar.abe_index = number
    
    '''Emission params'''
    emPar = exPar.copy()
    emPar.wl = 515 # emi wavelength [nm]
    
    '''Set-up params'''
    gridPar = psf.GridParameters()
    gridPar.Nz = nz
    gridPar.pxsizez = emPar.wl*exPar.n/(exPar.na**2) #DOF/2
    gridPar.Nx = 61
    gridPar.pxsizex = 50
    gridPar.N = 5
        
    grid_simul = GridFinder(gridPar)
    grid_simul.estimate(dset, exPar.wl, emPar.wl, emPar.na)
    grid_sample = ts.Sample()
        
    return (exPar, emPar, grid_simul,grid_sample, noise)

