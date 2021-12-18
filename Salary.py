# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 12:46:49 2020

@author: Alberto Barbado Gonzalez
"""

# Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# Data Preprocessing
# =============================================================================
# Cargar dataset
df = pd.read_csv('Salary_Data.csv')

# Visualizar los datos
df.plot(x='YearsExperience', y='Salary', title="Evolucion del Salario segun los Años de Experiencia") # Se ve relacion lineal

# Separacion en variables entrada/salida
X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


