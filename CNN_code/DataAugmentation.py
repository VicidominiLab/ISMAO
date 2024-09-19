# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:37:26 2024

@author: ffersini
"""

import Find_Correlogram as cr
import numpy as np
import tensorflow as tf
import random
import brighteyes_ism.dataio.mcs as mcs
import brighteyes_ism.analysis.Graph_lib as gra 
import random 

def center_crop(image, crop_height, crop_width):
    
    planes, img_height, img_width, channels = image.shape
    offset_height = (img_height - crop_height) // 2
    offset_width = (img_width - crop_width) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, crop_height, crop_width)
    
    return cropped_image


def random_center_crop(image, offset_x, offset_y, crop_height, crop_width):
    
    planes, img_height, img_width, channels = image.shape

    # Randomly choose center coordinates
    center_y = offset_x
    center_x = offset_y
    
    # Calculate the top left corner of the crop
    offset_height = center_y 
    offset_width = center_x 

    # Perform the cropping
    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)
    
    return cropped_image


def padding_image(img, target_height,target_width):
    
    planes, img_height, img_width, channels = img.shape

    # Calculate the amount of padding on each side
    pad_height = target_height - img_height
    pad_width = target_width - img_width
    
    # Calculate the padding offsets
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    
    # Pad the image with zeros (black)
    padded_image = np.pad(img, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    
    return padded_image


def data_augm(path,rep,factor):
           
    # dset,_ = mcs.load(path)
    # dset = np.squeeze(np.sum(dset, axis=-2))
    dset = np.load(path)
    sz = np.shape(dset)    
    dset_aug = np.zeros((sz[0]*rep,sz[1],sz[2]//factor,sz[3]//factor,sz[4]))
    k = 0
        
    for i in range(0,sz[0]):
    
        for h in range(0,rep):
            
            random_number = random.randrange(sz[2]//(factor),sz[2]//(factor)+sz[2]//(factor))
            print(random_number)
            img = np.squeeze(dset[i,:,:,:,:])
            print(img.shape)
            random_center_cropped_img = random_center_crop(img, random_number, random_number,sz[2]//(factor),sz[2]//(factor))
            print(random_center_cropped_img.shape)
            dset_aug[k,:,:,:,:] = random_center_cropped_img.numpy().astype("uint8")
            k+=1
    
    return dset_aug



