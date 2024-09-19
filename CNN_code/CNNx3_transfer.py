# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:56:32 2024

@author: ffersini
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pandas as pd 


#%% TransferLearning 

def transfer_cnnx3(path_weights,path_file,new_file,new_label,t_dset,t_labe):
        
    "Loaded model from disk"
    json_file = open(path_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(path_weights)
    print("Loaded model from disk")
    
    data = np.delete(new_file, 12, axis=3)
    
    #%% Launch
    print('TransferLearning')
    
    '''Launch'''
    
    learning_rate = 1e-4
    loaded_model.compile(optimizer= tf.optimizers.Adam(learning_rate), loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    s_e = int(np.shape(data)[0]) - int((np.shape(data)[0]) * 20 / 100)

    with tf.device('/CPU:0'):
        history = loaded_model.fit([data[0:s_e,:,:,:,0], data[0:s_e,:,:,:,1], data[0:s_e,:,:,:,2]],
                                    [new_label[0:s_e,:],new_label[0:s_e,:],new_label[0:s_e,:]], epochs=50, batch_size=8,shuffle = True,
                                    validation_data = ([data[s_e::,:,:,:,0],data[s_e::,:,:,:,1], data[s_e::,:,:,:,2]], [new_label[s_e::,:],new_label[s_e::,:],new_label[s_e::,:]]))
    
    print('Launch')
        
    #%% Save weights in TransferLearning folder
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    
    # or save to csv: 
    hist_csv_file = 'historyx3.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    #path = Add your path

    #model_json = loaded_model.to_json()
    #with open(r'path\modelx3_transfer.json', "w") as json_file:
    #    json_file.write(model_json)
        
    #loaded_model.save_weights(r'path\modelx3_transfer.weights.h5')
    #print("Saved model to disk")
    
    
    #%% Load weights in TransferLearning folder
    
    json_file = open(r'files\modelx3_transfer.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r'files\modelx3_transfer.weights.h5')
    print("Loaded model from disk")
        
    test = np.delete(t_dset, 12, axis=-2)
    
    with tf.device('/CPU:0'):
    
        pred_TL_x3 = loaded_model.predict([test[:,:,:,:,0],test[:,:,:,:,1],test[:,:,:,:,2]])
        pred_TL_x3 = np.round(pred_TL_x3,3)
    
    print('Launch')
    
    return pred_TL_x3,t_labe
