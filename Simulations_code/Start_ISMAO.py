# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:41:52 2024

@author: ffersini
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import Correlogram_Sim as cs

##Simulations

def random_Z():
    num_elements = np.random.randint(1, 8)  
    random_array = np.random.choice(np.arange(5, 12), size=num_elements, replace=False)  
    
    lower_bound = np.round(-np.pi/2, 2)
    upper_bound = np.round(np.pi/2, 2)
    
    corresponding_values = np.random.normal(loc=0, scale=1, size=num_elements) 
    corresponding_values = np.clip(corresponding_values, lower_bound, upper_bound)  
    
    corresponding_values = np.round(corresponding_values * 100) / 100  
    
    # Output the results
    print("Random Unique Array:", random_array)
    print("Corresponding Values:", corresponding_values)
    
    return random_array,corresponding_values


ite = 1  #iteration for simulations
nz = 1   #planes
var =  1 #type of sample

PSFs = np.zeros((ite,nz,61,61,25))
correlograms = np.zeros((ite,128,128,25,nz))
fp  = np.zeros((ite,25,3))
img  = np.zeros((ite,401,401,25,3))

for i in range(0,ite):
    number, amplitude = random_Z()
    PSFs[i,:,:,:,:], correlograms[i,:,:,:,:], fp[i,:,:], img[i,:,:,:,:]= cs.Corr_generator(amplitude, number, nz, var)

print(f'The random aberration selected is: number = {number}, amplitude = {amplitude}')


##

#import brighteyes_ism.analysis.Graph_lib as gra

#gra.ShowDataset(PSFs)

