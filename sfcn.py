
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
from tensorflow.keras.mixed_precision import LossScaleOptimizer
mixed_precision.set_policy(policy)

params = {
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)
    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)
    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)
    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)
    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)
    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)
    #block 7
    x=AveragePooling3D()(x)
    x=Dropout(.2)(x)
    x=Flatten(name="flat1")(x)
    x=Dense(units=1, activation='sigmoid',name="dense1")(x)


    return x

def create_model(lr):
    opt = Adam(lr=lr)
    opt = LossScaleOptimizer(opt, dynamic=True)
    metr = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
    inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
    z = sfcn([inputA])
    model = Model(inputs=[inputA], outputs=[z])
    model.summary()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metr)
    return model