# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:11:56 2022

@author: ASUS
"""
# Importing libraries
import os
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Convolution2D 
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D 
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import Dense 
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, auc, log_loss


def image_data_generator(data_dir="",
                         train_data=False,
                         batch_size=10,
                         target_size=(100, 100),
                         color_mode='rgb',
                         class_mode='binary',
                         shuffle=True):
    
    if train_data:
        datagen = ImageDataGenerator(rescale=1./255,
                                      rotation_range=20,
                                      width_shift_range=0.2,
                                      height_shift_range=0.2,
                                      shear_range=0.2,
                                      zoom_range=0.2,
                                      horizontal_flip=True,
                                     validation_split=0.2
                                     )
    else:
        datagen = ImageDataGenerator(rescale=1./255)
        
    generator = datagen.flow_from_directory(data_dir,
                                            target_size=target_size,
                                            color_mode=color_mode,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            class_mode=class_mode)
    return generator

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
    
    return model

batch_size = 10
height, width = (32, 32)
epochs = 20
color_mode = "grayscale"
optimizer = "rmsprop"
loss = "binary_crossentropy"
class_mode='binary'
output_classes=1 # Number of output classes
final_activation="sigmoid"
# Canales según el tipo de color_mode
if color_mode == "grayscale":
    channels = 1
    grayscale = True
else:
    channels = 3
    grayscale = False

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

test_set = image_data_generator('./Training/horse-or-human/test',
                     train_data=False,
                     batch_size=batch_size,
                     target_size=(height, width),
                     color_mode=color_mode,
                     class_mode=class_mode,
                     shuffle=True)

# Entrenamiento del modelo
model = build_model(optimizer=optimizer,
                loss=loss,
                height=height,
                width=width,
                channels=channels,
                output_classes=output_classes,
                final_activation=final_activation)
print(model.summary())
model.fit_generator(training_set,
 steps_per_epoch=batch_size,
 epochs=epochs,validation_data=val_set)
# Guardamos el modelo en un archivo binario
model.save('model_horses_vs_humans.h5')

# Model Evaluation

# Loss/Accuracy
score = model.evaluate(test_set, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Load test set and turn it into matrices arrays
path = './Training/horse-or-human/test'
entries = os.listdir(path)

X_test = []
y_test = []
for entry in entries:
    subpath = path +'/'+ entry
    files = []
    for _, _, f in os.walk(subpath):
        files += f

    X_test += [np.expand_dims(img_to_array(load_img(subpath + '/' + f,
                                                    target_size = (height, width),
                                                    grayscale=grayscale)), axis = 0) for f in files]
   
    if entry == "horses":
        y_test += [0]*len(files)
    else:
        y_test += [1]*len(files)

# Obtain predictions for all test set
y_pred = [model.predict_classes(x)[0][0] for x in X_test]

# Evaluate results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("Precision: ", np.round(precision_score(y_test, y_pred),4))
print("Recall: ", np.round(recall_score(y_test, y_pred),4))
print("f1_score: ", np.round(f1_score(y_test, y_pred),4))