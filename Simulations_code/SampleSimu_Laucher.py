# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 16:08:48 2024

@author: ffersini
"""

import SampleParam_Sim as sg
import random 
import numpy as np 
import tensorflow as tf

#%%Sample_Generator

def sample_generator(grid_samp,var):
    
    if var == 0:
        depthmap = sg.tubulin_gen(grid_samp.FOV,grid_samp.px_sample,15)
    elif var == 1:
        depthmap = sg.beads_gen(grid_samp.FOV,grid_samp.px_sample,random.randint(300,400), grid_samp.diameter)
    elif var == 2:
        depthmap = sg.mix_sample(grid_samp.FOV,grid_samp.px_sample,random.randint(300,400), grid_samp.diameter)
    else:
        depthmap = sg.mix_sample(grid_samp.FOV,grid_samp.px_sample,random.randint(450,550),grid_samp.diameter,iteration=20)
        
    counts = grid_samp.photon_flux * grid_samp.dwell_time
    depthmap *= counts
    tub_fft = tf.signal.rfft2d(depthmap, fft_length=[2*grid_samp.FOV-1, 2*grid_samp.FOV-1]) 
    
    return tub_fft, depthmap