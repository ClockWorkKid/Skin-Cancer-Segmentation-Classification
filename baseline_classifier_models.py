import math
import numpy as np
import pandas as pd
import gc
import keras
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cv2
import random

from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras import Model
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.layers import *
from keras.utils.vis_utils import plot_model
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.activations import sigmoid, tanh
import random
from tensorflow.keras.metrics import *
from tensorflow.keras.layers import *

from PIL import Image
from statistics import mean


def MobileNetClassifier(img_dims, class_type):

    model = tf.keras.applications.MobileNet(   
        include_top=False,
        weights="imagenet",
        input_shape=(img_dims, img_dims, 3),
        pooling=None,
    )

    # x = GlobalAveragePooling2D()(model.get_layer('conv4_block6_out').output) 
    x = GlobalAveragePooling2D()(model.output)
    x = Dense(512, 'relu')(x)
    x = Dense(128, 'relu')(x)
    if class_type=='binary':
        x = Dense(1, 'sigmoid')(x)
    else:
        x = Dense(7, 'softmax')(x)
        
    model = Model(model.input, x)
    
    return model


def ResNetClassifier(img_dims, class_type, type_="50"):

    if type_ == "50":
        model = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "101":
        model = tf.keras.applications.ResNet101(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "152":
        model = tf.keras.applications.ResNet152(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    else:
        raise TypeError("Model type not recognized")
    
    x = GlobalAveragePooling2D()(model.output)
    x = Dense(512, 'relu')(x)
    x = Dense(128, 'relu')(x)
    if class_type=='binary':
        x = Dense(1, 'sigmoid')(x)
    else:
        x = Dense(7, 'softmax')(x)
        
    model = Model(model.input, x)
    
    return model
    

def EfficientNetClassifier(img_dims, class_type, type_="B4", truncation=False):
    
    if type_ == "B0":      
        model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B1":
        model = tf.keras.applications.EfficientNetB1(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B2":
        model = tf.keras.applications.EfficientNetB2(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B3":
        model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B4":
        model = tf.keras.applications.EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B5":
        model = tf.keras.applications.EfficientNetB5(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B6":
        model = tf.keras.applications.EfficientNetB6(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_ == "B7":
        model = tf.keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    else:
        raise TypeError("Model type not recognized")

        
    if truncation:
        x = GlobalAveragePooling2D()(model.get_layer('block6a_expand_activation').output) #EffB2,B3, B4
    else:
        x = GlobalAveragePooling2D()(model.output)
    x = Dense(512, 'relu')(x)
    x = Dense(128, 'relu')(x)
    if class_type=='binary':
        x = Dense(1, 'sigmoid')(x)
    else:
        x = Dense(7, 'softmax')(x)

    return model


def DenseNetClassifier(img_dims, class_type, type_="121", truncation=False):

    if type_=="121":
        model = tf.keras.applications.DenseNet121( 
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_=="169":
        model = tf.keras.applications.DenseNet169(  
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    elif type_=="201"
       model = tf.keras.applications.DenseNet201( 
            include_top=False,
            weights="imagenet",
            input_shape=(img_dims, img_dims, 3),
            pooling=None,
        )
    else:
        raise TypeError("Model type not recognized")

    if truncation:
        x = GlobalAveragePooling2D()(model.get_layer('pool4_conv').output)
    else:
        x = GlobalAveragePooling2D()(model.output)
    x = Dense(512, 'relu')(x)
    x = Dense(128, 'relu')(x)
    if class_type=='binary':
        x = Dense(1, 'sigmoid')(x)
    else:
        x = Dense(7, 'softmax')(x)
        
    return model


def InceptionNetV3Classifier(img_dims, class_type, truncation=False):
    
    model = tf.keras.applications.InceptionV3(  
        include_top=False,
        weights="imagenet",
        input_shape=(img_dims, img_dims, 3),
        pooling=None,
    )
    
    if truncation:
        x = GlobalAveragePooling2D()(model.layers[-83].output) # InceptionNet
    else:
        x = GlobalAveragePooling2D()(model.output)
    x = Dense(512, 'relu')(x)
    x = Dense(128, 'relu')(x)
    if class_type=='binary':
        x = Dense(1, 'sigmoid')(x)
    else:
        x = Dense(7, 'softmax')(x)
        
    return model

