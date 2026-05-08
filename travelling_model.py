
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
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import argparse
import sys
import math
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

params = {
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

def train_travelling_model(studies, model, e, train_file, bs, seed):
    start_time = time.time()
    train = pd.read_csv(train_file)
    tm_studies = train[train['Study'].isin(studies)]
    np.random.seed(seed)  
    np.random.shuffle(studies)
    for s in studies:
        print("study-"+str(s))
        train_aux =  tm_studies[tm_studies['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        batch_size=bs
        if(len(IDs_list)<batch_size):
            batch_size=len(IDs_list)
        
        training_generator = DataGenerator(IDs_list,batch_size,(params['imagex'], params['imagey'], params['imagez']),True,train_file,'Group_bin')
        model.fit(training_generator, epochs=e,verbose=2)
    end_time = time.time()
    print(f"TM - Time: {end_time - start_time:.2f}s")
    return model

    