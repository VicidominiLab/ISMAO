# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:22:13 2024

@author: ffersini
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.models import model_from_json
import pandas as pd
import os

#%% Load_dset

def load_dset(cor_path,lab):
    
    cor = np.delete(cor_path[:,:,:,:,1], 12, axis=3)
    tr_img,  val_img, tr_lab, val_lab = train_test_split(cor, lab, test_size=0.15, random_state=3, shuffle=True)
    
    return tr_img,  val_img, tr_lab, val_lab

def CNNx1(cor_path,lab_path):
       
    tr_img,  val_img, tr_lab, val_lab = load_dset(cor_path,lab)
    
    #%% CNNx1 Structure
    
    model = keras.models.Sequential([
    
        keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(3,3), activation='relu', input_shape=(128,128,24)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
        
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
         
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(7)])
    
    model.summary()
    
    #%% Launcher CNN
    
    ephocs = 30
    learning_rate = 1e-4
    batch_size = 32
    
    model.compile(optimizer= tf.optimizers.Adam(learning_rate), loss='mse',
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    
    root_logdir = os.path.join(os.curdir, "logs\\fit\\")
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    with tf.device('/CPU:0'):
        history = model.fit(x = tr_img,    
                            y = tr_lab,
                            batch_size=batch_size,
                            epochs=ephocs,
                            verbose=1,
                            validation_split=None,
                            validation_data= ([val_img,val_lab]),
                            shuffle=True,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,
                            validation_batch_size=None,
                            validation_freq=1)
    
    #%% Save weights in Training folder
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # or save to csv: 
    hist_csv_file = 'history_x1.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # path = Add you path

    #model_json = model.to_json()
    #with open(r'path\modelx1.json', "w") as json_file:
    #    json_file.write(model_json)
    #model.save_weights(r'path\modelx1.weights.h5')
    #print("Saved model to disk")
    
    #return print("Saved model to disk")


def CNNx1_test(test_cor_path,labe_cor_path):
    
    json_file = open(r'files\modelx1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r'files\modelx1.weights.h5')
    print("Loaded model from disk")
     
    #%% Testing 
    
    dset = np.delete(test_cor_path[:,:,:,:,1], 12, axis=3)
    labe = labe_cor_path
    pred_TR_x1 = np.round(loaded_model.predict(dset),3)

    return pred_TR_x1,labe
