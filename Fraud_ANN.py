# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:25:48 2022

@author: ASUS
"""
# Dataset https://www.kaggle.com/ealaxi/paysim1

# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


# 1 Data Preparation
dataset = pd.read_csv('paysim_reduced.csv')

df_aux = pd.get_dummies(dataset['type']).astype('int')
dataset.drop(['type', 'Unnamed: 0', 'nameOrig',
 'nameDest', 'isFlaggedFraud', 'step',
 'newbalanceOrig', 'newbalanceDest'], axis=1, inplace=True)
dataset = dataset.join(df_aux)
X = dataset.loc[:, dataset.columns != 'isFraud'].values
y = dataset['isFraud'].values

# Train/Test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# 2 ANN Build 

# Importing the Keras libraries and packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense
from tf.keras.optimizers import SGD

"""
 Funcion para crear una NN para clasificacion binaria usando 2 HL

"""
def create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum,decay):

     # Initialising the ANN
     model = Sequential()
     # First HL
     # [batch_size x n_features] x [n_features x w_in]
     model.add(Dense(units = w_in, input_dim = n_features,
     kernel_initializer = 'normal',
     activation = 'relu'))
     # Second HL
     # [batch_size x w_in] x [w_in x w_h1]
     model.add(Dense(units = w_h1, input_dim = w_in,
     kernel_initializer = 'normal',
     activation = 'relu'))
    
     # Output Layer
     
     # [batch_size x w_h1] x [w_h1 x w_out]
     model.add(Dense(units = n_var_out, kernel_initializer = 'normal',activation = 'sigmoid'))
     # Compile Model
     # Loss Function -> Cross Entropy (Binary)
     # Optimizer -> sgd, adam...
     if optimizer == 'sgd':
         tf.keras.optimizers.SGD(lr=lr, momentum=momentum, decay=decay,nesterov=False)
         model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
     else:
         model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
     return model
 
# Parametros
n_features = np.shape(X_train)[1]
w_in = 12
w_h1 = 8
n_var_out = 1
batch_size = 100
nb_epochs = 100
optimizer = 'adam'
lr = 0.1
momentum = 0.01
decay = 0.0
# Create NN
model = create_nn(n_features, w_in, w_h1, n_var_out, optimizer, lr, momentum,
decay)

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = nb_epochs)


# 3 ANN Predictions

# Predict
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
