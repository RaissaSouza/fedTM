
from numpy.random import seed
seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import random
random.seed(42)
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
from fed_prox import train_fed_prox
from fed_avg import train_fed_avg
from travelling_model import train_travelling_model
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



#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-fn_train', type=str, help='training set')
parser.add_argument('-cycles', type=int, help='number of cycles')
parser.add_argument('-epochs_f', type=int, help='number of local epochs per cycle fed')
parser.add_argument('-epochs_t', type=int, help='number of local epochs per cycle tm')
parser.add_argument('-batch_size', type=int, help='batch size')
parser.add_argument('-split', type=int, help='split')
parser.add_argument('-wp', type=int, help='warm up')
parser.add_argument('-mu', type=float, help='mu')
parser.add_argument('-s', type=int, help='strategy')
parser.add_argument('-out', type=str, help='output filename')
args = parser.parse_args()


SEED=1
CYCLES = args.cycles
EPOCHS_F = args.epochs_f
EPOCHS_T = args.epochs_t
BATCH_SIZE = args.batch_size
LEARNING_RATE = 0.0001
TRAIN_FILE = args.fn_train
SPLIT = args.split
WM = args.wp
FN = args.out
MU = args.mu
STRATEGY = args.s

train = pd.read_csv(TRAIN_FILE)
all_studies =  train['Study'].unique()

study_counts = train['Study'].value_counts()

# Create two lists based on the counts
selected_studies = study_counts[study_counts >= SPLIT].index.tolist()
studies_not_selected = study_counts[study_counts < SPLIT].index.tolist()

# Define the studies to move
to_move = {"SALD", "UOA", "OASIS"} #because they only have one class

# Remove from selected_studies if present
selected_studies = [s for s in selected_studies if s not in to_move]

# Add to studies_not_selected if not already present
for study in to_move:
    if study not in studies_not_selected:
        studies_not_selected.append(study)

# Print results
print("Studies selected for warmup:", selected_studies)
print("Studies NOT selected:", studies_not_selected)

global_model = create_model(LEARNING_RATE)
global_model.save("global_model.h5")


for c in range(CYCLES):
    start_time = time.time()
    if STRATEGY==1:
        ### STRATEGY ONE FEDAVG Warm up with fed and include all centers in TM ###
        if(c<WM):
            global_model_weights = global_model.get_weights()
            optimizer_weights = global_model.optimizer.get_weights()
        
    
            del global_model
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            
            update_weights = train_fed_avg(selected_studies, LEARNING_RATE, global_model_weights, optimizer_weights, EPOCHS_F, TRAIN_FILE, BATCH_SIZE)
            #update_weights = train_fed_avg(all_studies, LEARNING_RATE, global_model_weights, optimizer_weights, EPOCHS_F, TRAIN_FILE, BATCH_SIZE) # use this for the baseline training and pass WM = 30 
            global_model = tf.keras.models.load_model("global_model.h5")
            global_model.save(FN+"_"+str(args.cycles)+"_"+str(args.epochs_f)+"_"+str(args.epochs_t)+"_"+str(c)+".h5") #used to sabe baseline when just FL is trained
            global_model.set_weights(update_weights)

        else:
            if(c==WM):
                global_model = tf.keras.models.load_model("global_model.h5")
            global_model = train_travelling_model(all_studies, global_model, EPOCHS_T, TRAIN_FILE, BATCH_SIZE, c+SEED)
            global_model.save(FN+"_"+str(args.cycles)+"_"+str(args.epochs_f)+"_"+str(args.epochs_t)+"_"+str(c)+".h5")
    elif STRATEGY==2:
        print("strategy 2")
         ### STRATEGY TWO FEDPROX Warm up with fed and include all centers in TM ###
        if(c<WM):
            global_model_weights = global_model.get_weights()
            optimizer_weights = global_model.optimizer.get_weights()
        
    
            del global_model
            K.clear_session()
            tf.keras.backend.clear_session()
            gc.collect()
            
            update_weights = train_fed_prox(selected_studies, LEARNING_RATE, global_model_weights, optimizer_weights, EPOCHS_F, TRAIN_FILE, BATCH_SIZE, MU)
            #update_weights = train_fed_prox(all_studies, LEARNING_RATE, global_model_weights, optimizer_weights, EPOCHS_F, TRAIN_FILE, BATCH_SIZE, MU) #use this for baseline with WM =30
            global_model = tf.keras.models.load_model("global_model.h5")
            global_model.save(FN+"_"+str(args.cycles)+"_"+str(args.epochs_f)+"_"+str(args.epochs_t)+"_"+str(c)+".h5") #used to sabe baseline when just FL is trained
            global_model.set_weights(update_weights)

        else:
            if(c==WM):
                global_model = tf.keras.models.load_model("global_model.h5")
            global_model = train_travelling_model(all_studies, global_model, EPOCHS_T, TRAIN_FILE, BATCH_SIZE, c+SEED)
            global_model.save(FN+"_"+str(args.cycles)+"_"+str(args.epochs_f)+"_"+str(args.epochs_t)+"_"+str(c)+".h5")

    # Implement learning rate decay
    LEARNING_RATE = LEARNING_RATE * tf.math.exp(-0.1)
    print(f"Learning rate after decay for cycle {c + 1}: {LEARNING_RATE}")
    
    # Update optimizer with the new learning rate
    K.set_value(global_model.optimizer.learning_rate, LEARNING_RATE)

    global_model.save("global_model.h5")


    end_time = time.time()
    print(f"Cycle {c+1} - Time: {end_time - start_time:.2f}s")
