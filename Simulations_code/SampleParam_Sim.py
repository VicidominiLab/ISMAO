# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:29:24 2024

@author: ffersini
"""

import numpy as np
import random
import SimulationTubulin as simTub
import matplotlib.pyplot as plt
   
def single_beads(FOV,px,num,diam):
       
    nx, ny = (FOV,FOV)
    x = np.linspace(0, FOV, nx)
    y = np.linspace(0, FOV, ny)
    xv, yv = np.meshgrid(x, y)
    
    radius = (diam/2)/px
    
    x, y = np.meshgrid(x, y)
    beads = np.zeros((FOV,FOV))
                        
    x0 = FOV//2
    y0 = FOV//2
    mask = np.zeros((FOV,FOV))

    s_x = [int(x0-radius),int(x0+radius)]
    s_y = [int(y0-radius),int(y0+radius)]
    
    try:
        mask[s_x[0]:s_x[1],s_y[0]:s_y[1]] += 1
    except:
        pass
               
    beads += mask
     
    beads[beads>0] = 1
        
    return beads

def beads_gen(FOV,px,num,diam):
       
    nx, ny = (FOV,FOV)
    x = np.linspace(0, FOV, nx)
    y = np.linspace(0, FOV, ny)
    xv, yv = np.meshgrid(x, y)
    
    radius = (diam/2)/px
    
    x, y = np.meshgrid(x, y)
    beads = np.zeros((FOV,FOV))
            
    for i in range(0,num):
            
        x0 = random.randint(0,FOV)
        y0 = random.randint(0,FOV)
        mask = np.zeros((FOV,FOV))
    
        s_x = [int(x0-radius),int(x0+radius)]
        s_y = [int(y0-radius),int(y0+radius)]
        
        try:
            mask[s_x[0]:s_x[1],s_y[0]:s_y[1]] += 1
        except:
            pass
                   
        beads += mask
     
    beads[beads>0] = 1
        
    return beads

#%%

def tubulin_gen(FOV,pxsize,iteration):
        
    phTub_sim = np.zeros((FOV,FOV))

    for i in range(0,iteration):
        
        tubulin = simTub.tubSettings()
        tubulin.xy_pixel_size = pxsize
        tubulin.xy_dimension = FOV
        tubulin.xz_dimension = 1    
        tubulin.z_pixel = 1   
        tubulin.n_filament = 9#random.randint(10, 20)
        tubulin.radius_filament = random.randint(50,80) #nm
        tubulin.intensity_filament = [0.5,1]  
        phTub = simTub.functionPhTub(tubulin)
        phTub_sim = phTub[:,:,0]
        phTub_sim/=np.max(phTub_sim)
    
    return phTub_sim

#%%

def mix_sample(FOV,pxsize,num,diam,iteration=10):
    
    beads_map = beads_gen(FOV,pxsize,num,diam)
    tubul_map = tubulin_gen(FOV,pxsize,iteration)
    sum_map = beads_map + tubul_map
    sum_map[sum_map>0] = 1
    
    return sum_map
    
