# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:11:56 2022

@author: ASUS
"""
# Importing libraries
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


from tensorflow import keras
from tensorflow.keras.models import Sequential # Para inicializar la NN (como es una Secuencia de layers, lo hago igual que con ANN; no uso la inici. de Graph)
from tensorflow.keras.layers import Convolution2D # Para hacer el paso de convolución, 1er step
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D # Para el Pooling step, paso 2
from tensorflow.keras.layers import Flatten # Para el flattening, step 3
from tensorflow.keras.layers import Dense # Para añadir los fully-connected layers hacia el layer de outputs
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

def build_model(optimizer="adam",
 loss='binary_crossentropy',
 height=32,
 width=32,
 channels=1,
 output_classes=1,
 final_activation="sigmoid"):
    
# Inicialización de la CNN
    model=Sequential()
    
    # Paso 1 - 1a Convolución
    # En Convolution: nº filtros, filas, columnas. 
    model.add(Convolution2D(filters=6,
                            kernel_size=(3, 3), 
                            activation='relu',
                            input_shape=(height,width,channels)))
    
    # Paso 2 - 1er Avg. Pooling
    model.add(AveragePooling2D(pool_size=(2, 2),
                                   strides=2))
    
    # Paso 3 - 2nda Convolución
    model.add(Convolution2D(filters=16,
                            kernel_size=(3, 3),
                            activation='relu'))
    
    # Paso 4 - 2ndo Avg. Pooling
    model.add(AveragePooling2D(pool_size=(2, 2),
                                   strides=2))
    
    # Paso 5 - Flattening
    model.add(Flatten())
    
    # RED NN FULLY CONNECTED 
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units = output_classes, activation = final_activation))
    model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])
    
    
    batch_size = 20
    height, width = (32, 32)
    epochs = 80
    color_mode = "rgb"
    optimizer = "adam"
    loss = "binary_crossentropy"
    class_mode='binary'
    output_classes=1 # Number of output classes
    final_activation="sigmoid"
    return model
    
training_set = image_data_generator('./Training/horse-or-human/train',
 train_data=True,
 batch_size=batch_size,
 target_size=(height, width),
 color_mode=color_mode,
 class_mode=class_mode,
 shuffle=True)
val_set = image_data_generator('./Training/horse-or-human/validation',
 train_data=False,
 batch_size=batch_size,
 target_size=(height, width),
 color_mode=color_mode,
 class_mode=class_mode,
 shuffle=True)
# Definición del modelo y visualización de la arquitectura definida.
model = build_model(optimizer=optimizer,
 loss=loss,
 height=height,
 width=width,
 channels=channels,
 output_classes=output_classes,
 final_activation=final_activation)
print(model.summary())
# Hago el fit de los sets de datos al modelo y entrenamiento del mismo
model.fit_generator(training_set,
 steps_per_epoch=batch_size,
 epochs=epochs,validation_data=val_set)
# Guardamos el modelo en un archivo binario
model.save('model_horses_vs_humans.h5')