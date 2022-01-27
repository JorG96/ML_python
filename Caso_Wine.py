# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:43:23 2022

@author: ASUS
"""
# 1. ANALISIS EXPLORATORIO
#   Librerias
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar Datos
df = pd.read_csv("Caso_Wine.csv")
print(df.columns)
# Identificacion de variables
df.info()
df.describe()
# Grafico de densidad
fig, ax = plt.subplots(5,3, figsize=(14,12))
axes_ = [axes_row for axes in ax for axes_row in axes]
for i,c in enumerate(df.columns):
    sns.distplot(df[c], ax = axes_[i], color = 'Set1')
    plt.tight_layout()
# Graficos de caja
fig, ax = plt.subplots(5,3, figsize=(14,12))
axes_ = [axes_row for axes in ax for axes_row in axes]
for i,c in enumerate(df.columns):
    sns.boxplot(df[c], ax = axes_[i], palette='Set1')
    plt.tight_layout()
# Matriz de correlacion
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(method='spearman'),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
# 2. MODELO DE CLUSTERING
# Columnas


# 3. MODELO PREDICTIVO

# PCA
