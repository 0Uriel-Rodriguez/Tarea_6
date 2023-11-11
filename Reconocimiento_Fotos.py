#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import datetime
import pathlib
import os
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import wandb
from wandb.keras import WandbCallback
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import RMSprop, Adam, Adamax

batch_size = 20
img_height = 180
img_width = 180
num_class = 2 
epochs = 30 

train_dir = r'Dataset\train_set' #directorio de entrenamiento
test_dir = r'Dataset\val_set' #directorio de prueba

train_datagen = ImageDataGenerator(  
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range = 5,
    horizontal_flip=True)

train = train_datagen.flow_from_directory(  
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test = test_datagen.flow_from_directory(  
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

#Para cargar la red pre_entrenada:
pre_trained_model = tf.keras.models.load_model('Prueba_1.h5')
########################################
wandb.init(project="reconocimiento facial")
wandb.config.epochs = epochs
wandb.config.batch_size = batch_size
wandb.config.optimizer = optimizer
#######################################

model = tf.keras.Sequential()
model.add(pre_trained_model.layers[0])
model.add(pre_trained_model.layers[1])
model.add(pre_trained_model.layers[2])
model.add(pre_trained_model.layers[3])
model.add(pre_trained_model.layers[4])
model.add(pre_trained_model.layers[5])
model.add(pre_trained_model.layers[6])
model.add(pre_trained_model.layers[7])
model.add(pre_trained_model.layers[8])
model.add(Dense(113, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

for layer in model.layers[:8]:
    layer.trainable = False

model.summary()


model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

log_dir="Graph/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
#python -m tensorboard.main --logdir=/Graph  <- Para correr Tensor board
#tensorboard  --logdir Graph/

model.fit(
    train,
    batch_size=batch_size,
    epochs=15,
    verbose=1,
    validation_data=test,
    callbacks= [WandbCallback()])

#Para guardar el modelo en disco
model.save("Prueba_fotos.h5")

