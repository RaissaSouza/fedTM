import tensorflow as tf
import csv
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
from sfcn import create_model
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse
import sys
import math
import time
import gc
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
from tensorflow.keras.mixed_precision import LossScaleOptimizer



params = {
    'imagex': 160,
    'imagey': 192,
    'imagez': 160
}

def train_fed_avg(studies, lr, global_weights, optimizer_weights, e, train_file, bs):
    start_time = time.time()
    client_weights = []
    train = pd.read_csv(train_file)
    train_fed = train[train['Study'].isin(studies)]
    
    for s in studies:
        start_time_s = time.time()
        print(f"Training on site: {s}")
        train_aux = train_fed[train_fed['Study'] == s]
        IDs_list = train_aux['Subject'].to_numpy()
        
        # Reset session to free up memory
        K.clear_session()
        gc.collect()
        
        # Create local model and set global weights
        local_model = create_model(lr)
        local_model.set_weights(global_weights)
        
        # Data generator
        training_generator = DataGenerator(
            IDs_list, bs,
            (params['imagex'], params['imagey'], params['imagez']),
            True, train_file, 'Group_bin'
        )
        
        # Train model using model.fit()
        local_model.fit(training_generator, epochs=e, verbose=1)
        
        # Collect trained weights
        client_weights.append(local_model.get_weights())

        end_time_s = time.time()
        print(f"Study - Time: {end_time_s - start_time_s:.2f}s")
        
        # Cleanup after processing this study
        del local_model
        gc.collect()
    
    # FedAvg aggregation
    new_global_weights = [tf.reduce_mean(tf.stack(layer_weights), axis=0) for layer_weights in zip(*client_weights)]
    end_time = time.time()
    print(f"FedAVG - Time: {end_time - start_time:.2f}s")
    return new_global_weights
