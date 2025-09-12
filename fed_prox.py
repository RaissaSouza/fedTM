
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

# FedProx Parameters
#MU = 0.025  # Proximal term coefficient


# # Define the FedProx loss function
# def fedprox_loss(global_weights, local_model, x_batch, y_batch, MU):
#     # Forward pass: compute the predicted labels
#     y_pred = local_model(x_batch, training=True)


#     y_pred = tf.reshape(y_pred, [-1])  # Flatten if needed for binary classification
#     y_batch = tf.reshape(y_batch, [-1])  # Flatten if needed for binary classification
   
    
#     # Compute the base loss (binary cross-entropy)
#     base_loss = tf.keras.losses.binary_crossentropy(y_batch, y_pred)
    
#     # Compute the FedProx regularization term (proximal loss)
#     prox_loss = 0.0
#     for gw, lw in zip(global_weights, local_model.get_weights()):
#         # Calculate the L2 norm between local weights and global weights
        
#         lw = tf.cast(tf.reshape(lw, [-1]), tf.float16)
#         gw = tf.cast(tf.reshape(gw, [-1]), tf.float16)

#         prox_loss += tf.reduce_sum(tf.norm(lw - gw, ord='euclidean'))
    
#     # Multiply by the regularization parameter mu
#     prox_loss *= MU
    
#     # Return the total loss (base loss + prox_loss)
#     return tf.reduce_mean(base_loss) + tf.cast(prox_loss, base_loss.dtype)


# def train_fed_prox(studies, lr, global_weights, optimizer_weights, e, train_file, bs, mu):
#     start_time = time.time()
#     client_weights = []
#     train = pd.read_csv(train_file)
#     train_fed = train[train['Study'].isin(studies)]
  

#     for s in studies:
#         print(f"Training on site: {s}")
#         train_aux =  train_fed[train_fed['Study']==s]
#         IDs_list = train_aux['Subject'].to_numpy()

#         # Reset session to free up memory
#         K.clear_session()
#         tf.keras.backend.clear_session()
#         gc.collect()

#         # Create local model and set weights
#         local_model = create_model(lr)
#         local_model.set_weights(global_weights)
#         #local_model.optimizer.set_weights(optimizer_weights)

#         # Data generator
#         training_generator = DataGenerator(
#             IDs_list, bs, 
#             (params['imagex'], params['imagey'], params['imagez']), 
#             True, train_file, 'Group_bin'
#         )

# # Training loop
#         for epoch in range(e):
#             start_time = time.time()
#             print(f"Epoch {epoch+1}/{e}")

#             epoch_loss = []
#             accuracy_metric = tf.keras.metrics.BinaryAccuracy()

#             for x_batch, y_batch in training_generator:
#                 with tf.GradientTape() as tape:
#                     loss = fedprox_loss(global_weights, local_model, x_batch, y_batch,mu)

#                 gradients = tape.gradient(loss, local_model.trainable_variables)
#                 local_model.optimizer.apply_gradients(zip(gradients, local_model.trainable_variables))

#                 # Compute accuracy
#                 y_pred = local_model(x_batch, training=False)
#                 accuracy_metric.update_state(y_batch, y_pred)

#                 epoch_loss.append(loss.numpy())

#             end_time = time.time()
#             avg_loss = np.mean(epoch_loss)
#             accuracy = accuracy_metric.result().numpy()

#             print(f"Epoch {epoch+1} - Time: {end_time - start_time:.2f}s - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")


#         # Collect trained weights
#         client_weights.append(local_model.get_weights())

#         # Cleanup after processing this study
#         del local_model
#         gc.collect()


#     # Efficient FedAvg aggregation
#     #print(client_weights)
#     new_global_weights = [tf.reduce_mean(tf.stack(layer_weights), axis=0) for layer_weights in zip(*client_weights)]
#     end_time = time.time()
#     print(f"FedProx - Time: {end_time - start_time:.2f}s")
#     return new_global_weights

def fedprox_loss(mu, global_weights, local_model):
    print(mu)
    def loss(y_true, y_pred):
        base_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Convert global weights to TF tensors
        global_weights_tensors = [tf.convert_to_tensor(gw, dtype=lw.dtype) for lw, gw in zip(local_model.weights, global_weights)]

        # Compute FedProx regularization
        prox_loss = 0.0
        for gw, lw in zip(global_weights_tensors, local_model.weights):
            lw = tf.cast(tf.reshape(lw, [-1]), tf.float16)
            gw = tf.cast(tf.reshape(gw, [-1]), tf.float16)
            prox_loss += tf.reduce_sum(tf.norm(lw - gw, ord='euclidean'))

        prox_loss *= mu
        return tf.reduce_mean(base_loss) + tf.cast(prox_loss, base_loss.dtype)

    return loss

def compute_weight_difference(global_weights, local_model):
    """Compute the Euclidean norm difference between global and local weights."""
    # Convert global weights to TF tensors
    global_weights_tensors = [tf.convert_to_tensor(gw, dtype=lw.dtype) for lw, gw in zip(local_model.weights, global_weights)]
    norm_diffs = []
    for gw, lw in zip(global_weights_tensors, local_model.weights):
        diff = np.linalg.norm(lw - gw)
        norm_diffs.append(diff)
    return np.mean(norm_diffs)  # Return the average weight difference


def train_fed_prox(studies, lr, global_weights, optimizer_weights, e, train_file, bs, mu):
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
        tf.keras.backend.clear_session()
        gc.collect()

        # Create local model and set weights
        local_model = create_model(lr)
        local_model.set_weights(global_weights)

        local_model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=fedprox_loss(mu, global_weights, local_model),
            metrics=['accuracy']
        )

        # Data generator
        training_generator = DataGenerator(
            IDs_list, bs,
            (params['imagex'], params['imagey'], params['imagez']),
            True, train_file, 'Group_bin'
        )

        # Before training
        initial_diff = compute_weight_difference(global_weights, local_model)
        print(f"Initial weight difference: {initial_diff}")

        # Train model using fit()
        local_model.fit(training_generator, epochs=e, verbose=1)

        # After training
        final_diff = compute_weight_difference(global_weights, local_model)
        print(f"Final weight difference: {final_diff}")

        # Collect trained weights
        client_weights.append(local_model.get_weights())

        end_time_s = time.time()
        print(f"Study - Time: {end_time_s - start_time_s:.2f}s")

        # Cleanup after processing this study
        del local_model
        gc.collect()

    # Aggregate weights across clients
    new_global_weights = [tf.reduce_mean(tf.stack(layer_weights), axis=0) for layer_weights in zip(*client_weights)]
    end_time = time.time()
    print(f"FedProx - Time: {end_time - start_time:.2f}s")
    return new_global_weights

