# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:07:26 2022

@author: ASUS
"""
 # LeNet Architecture
# Importar librearias
from tensorflow.keras.models import Sequential # Para inicializar la NN (como es una Secuencia de layers, lo hago igual que con ANN; no uso la inici. de Graph)
from tensorflow.keras.layers import Convolution2D # Para hacer el paso de convolución
from tensorflow.keras.layers import AveragePooling2D # Para el Pooling step
from tensorflow.keras.layers import Flatten # Para el flattening
from tensorflow.keras.layers import Dense # Para añadir los fully-connected layers hacia el layer de outputs

# Extraccion de CNN

# Paso 1 - 1a Convolución


# Paso 2 - 1er Avg. Pooling

# Paso 3 - 2nda Convolución

# Paso 4 - 2ndo Avg. Pooling

# Paso 5 - Flattening