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

# 3 ANN Predictions