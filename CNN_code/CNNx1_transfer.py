# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:33:39 2024

@author: ffersini
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pandas as pd 


#%% TransferLearning 

def transfer_cnnx1(path_weights,path_file,new_file,new_label,t_dset,t_labe):
    
    "Loaded model from disk"
    json_file = open(path_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    
    data = np.delete(np.load(new_file), 12, axis=3)[:,:,:,:,1]
    labe = np.load(new_label)
    
    print('TransferLearning')
    
    '''Launch'''
    
    learning_rate = 1e-4
    loaded_model.compile(optimizer= tf.optimizers.Adam(learning_rate), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    s_e = int(np.shape(data)[0]) - int((np.shape(data)[0])*20/100)
    with tf.device('/CPU:0'):
        history = loaded_model.fit(data[0:s_e,:,:,:], labe[0:s_e,:], epochs=30, batch_size=16, shuffle = True, validation_data = (data[s_e::,:,:,:], labe[s_e::,:]))
    
    print('Launch')
    
    '''Save weights in TransferLearning folder'''

    hist_df = pd.DataFrame(history.history) 
    
    # or save to csv: 
    hist_csv_file = 'historyx3.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    # path = Add you path

   # model_json = loaded_model.to_json()
   # with open(r'path\modelx1_transfer.json', "w") as json_file:
    #    json_file.write(model_json)
        
   # loaded_model.save_weights(r'path\modelx1_transfer.weights.h5')
   # print("Saved model to disk")
    
    
    #%% Load weights in TransferLearning folder
    
    json_file = open(r'files\modelx1_transfer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r'files\modelx1_transfer.weights.h5')
    print("Loaded model from disk")
    
    #%% Testing

    dset = np.delete(np.load(t_dset)[:,:,:,:,1], 12, axis=3)
    labe = np.load(t_labe)
    
    with tf.device('/CPU:0'):
        pred_TL_x1 = loaded_model.predict(dset)
        pred_TL_x1 = np.round(pred_TL_x1,3)
    
    print('Testing')
    
    return pred_TL_x1,labe
   
