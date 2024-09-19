# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:22:05 2024

@author: ffersini
"""

import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, BatchNormalization,Dropout
from keras.models import Model
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
import os


#%% CNNx3 structure

def load_dset(cor_path,lab):
    
    cor = np.delete(cor_path, 12, axis=3)
    tr_img,  val_img, tr_lab, val_lab = train_test_split(cor, lab, test_size=0.15, random_state=3, shuffle=True)
    
    return tr_img,  val_img, tr_lab, val_lab


def CNNx3(cor_path,lab):
    
    tr_img,  val_img, tr_lab, val_lab = load_dset(cor_path,lab_path)

    digit_a = Input(shape=(128,128,24))
    x = Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu', input_shape=(128,128,24))(digit_a)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    out_a = Flatten()(x)
    
    digit_b = Input(shape=(128,128,24))
    x = Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu', input_shape=(128,128,24))(digit_b)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    out_b = Flatten()(x)
    
    digit_c = Input(shape=(128,128,24))
    x = Conv2D(filters=16, kernel_size=(5,5), strides=(3,3), activation='relu', input_shape=(128,128,24))(digit_c)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    out_c = Flatten()(x)
    
    concatenated = concatenate([out_a, out_b, out_c])
    out = Dense(1024, activation='relu')(concatenated)
    out = Dropout(0.5)(out)
    out = Dense(1024, activation='relu')(out)
    out = Dropout(0.5)(out)
    out = Dense(7)(out)
       
    
    model = Model([digit_a, digit_b, digit_c], out)
    print(model.summary())
    #plot_model(model, "multi_input_and_output_model_128x128.png", show_shapes=True)
    
    ephocs = 30
    learning_rate = 1e-4
    batch_size = 64
    
    model.compile(optimizer= tf.optimizers.Adam(learning_rate), loss='mse',
                metrics=[tf.keras.metrics.MeanSquaredError()])

    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    
    root_logdir = os.path.join(os.curdir, "logs\\fit\\")
    run_logdir = get_run_logdir()
    tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
       
    with tf.device('/CPU:0'):
        history = model.fit(
            x = ([tr_img[:,:,:,:,0],tr_img[:,:,:,:,1],tr_img[:,:,:,:,2]]),
            y = ([tr_lab, tr_lab,tr_lab]),
            batch_size=batch_size,
            epochs=ephocs,
            verbose=1,
            validation_data=([[val_img[:,:,:,:,0],val_img[:,:,:,:,1],val_img[:,:,:,:,2]],[val_lab,val_lab,val_lab]]),
            shuffle=True,
            initial_epoch=0,
            validation_freq=1)
        
    #%% Save weights in Training folder    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # or save to csv: 
    hist_csv_file = 'historyx3.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # path = Add you path


    #model_json = model.to_json()
    #with open(r'path\modelx3.json', "w") as json_file:
    #    json_file.write(model_json)
        
    #model.save_weights(r'path\modelx3.weights.h5')
    
    #return print("Saved model to disk")

#%% Load weights from Training folder○

def CNNx3_test(test_cor_path,labe_cor_path):

    json_file = open(r'files\modelx3.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r'files\modelx3.weights.h5')
    print("Loaded model from disk")
     
    #%% Testing
    
    dset = np.delete(test_cor_path, 12, axis=3)
   
    with tf.device('/CPU:0'):
        
        pred_TR_x3 = loaded_model.predict([dset[:,:,:,:,0],dset[:,:,:,:,1],dset[:,:,:,:,2]])
        pred_TR_x3 = np.round(pred_TR_x3,3)
    
    return pred_TR_x3,labe_cor_path
