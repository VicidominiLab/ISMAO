# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:20:58 2024

@author: ffersini
"""

import copy as cp

class Sample():
    
    def __init__(self, iteration = 1, fp_norm = True,
                 sample = True, diameter = 100, photon_flux = 40*1e6,
                 dwell_time  = 50*1e-6, FOV= 401, crp = 128, px_sample = 50):
        
        self.iteration = iteration
        self.fp_norm = fp_norm
        self.sample = sample
        self.diameter = diameter
        self.photon_flux = photon_flux # Hz
        self.dwell_time  = dwell_time # s
        self.FOV = FOV
        self.crp = crp
        self.px_sample = px_sample

        
    def copy(self):
        return cp.copy(self)

    def Print(self):
        dic = self.__dict__
        names = list(dic)
        values = list(dic.values())
        for n, name in enumerate(names):
            print(name, end = '')
            print(' ' * int(14 - len(name)), end = '')
            print("" if values[n] is None else f'{values[n]:.2f}')