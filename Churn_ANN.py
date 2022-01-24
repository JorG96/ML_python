# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:19:55 2022

@author: ASUS
"""
#1. CARGA DE DATASET
# Librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns # For creating plots
import matplotlib.ticker as mtick # For specifying the axes tick format 
from sklearn.metrics import roc_curve , auc
from sklearn.metrics import confusion_matrix, classification_report
dataset = pd.read_csv('TELCO_U4.csv')
# Exploracion inicial
dataset.head()
# Identificación de variables
list_cont = ['tenure', 'MonthlyCharges', 'TotalCharges']
list_ord = ['PhoneService', 'Dependents', 'Contract', 'InternetService', 'PaperlessBilling', 'Churn']
list_not_ord = [x for x in list(dataset.columns) if x not in list_cont + list_ord and x != 'customerID']
#2. ANALISIS EXPLORATORIO
# Información relevante sobre las columnas
dataset.info()
# Pie charts
f, ax = plt.subplots(figsize=(8, 10))

# Género
plt.subplot(2, 2, 1)
plt.title("Género Masculino vs Femenino")
feature_used = "gender"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# Partner
plt.subplot(2, 2, 2)
plt.title("Cliente tiene o no Compañero")
feature_used = "Partner"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')


# Dependents
plt.subplot(2, 2, 3)
plt.title("Cliente tiene o no Dependientes")
feature_used = "Dependents"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# InternetService
plt.subplot(2, 2, 4)
plt.title("Servicio de Internet")
feature_used = "InternetService"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

plt.show()

# Histogramas charts
f, ax = plt.subplots(figsize=(8, 10))

# tenure
plt.subplot(2, 1, 1)
plt.hist(dataset['tenure'], color = 'red')
plt.title('Histograma Meses de Permanencia como Cliente')
plt.xlabel('Meses con la Compañia')

plt.show()

f, ax = plt.subplots(figsize=(8, 10))

# SeniorCitizen
plt.subplot(2, 2, 1)
plt.title('Clientes Senior (Tercera Edad)')
feature_used = "SeniorCitizen"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# Whether the customer is a senior citizen or not (1, 0)
plt.show()
# Pie charts
f, ax = plt.subplots(2, 2, figsize=(10, 10))

# PhoneService
plt.subplot(2, 2, 1)
plt.title("Servicio Telefónico")
feature_used = "PhoneService"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# OnlineSecurity
plt.subplot(2, 2, 2)
plt.title("Servicio de Seguridad En Línea")
feature_used = "OnlineSecurity"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# MultipleLines
plt.subplot(2, 2, 3)
plt.title("Servicio de Varias Líneas Telefónicas")
feature_used = "MultipleLines"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# OnlineBackup
plt.subplot(2, 2, 4)
plt.title("Servicio de Respaldo Datos En Línea")
feature_used = "OnlineBackup"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')
  
plt.show()
# Pie charts
f, ax = plt.subplots(2, 2, figsize=(10, 10))

# DeviceProtection
plt.subplot(2, 2, 1)
plt.title("Servicio de Protección de Dispositivo")
feature_used = "DeviceProtection"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# TechSupport
plt.subplot(2, 2, 2)
plt.title("Servicio de Soporte Téccnico")
feature_used = "TechSupport"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# StreamingTV
plt.subplot(2, 2, 3)
plt.title("Servicio de Streaming TV")
feature_used = "StreamingTV"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# StreamingMovies
plt.subplot(2, 2, 4)
plt.title("Servicio de Peliculas Streaming")
feature_used = "StreamingMovies"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

plt.show()
# Pie charts
f, ax = plt.subplots(2, 1, figsize=(10, 10))

# Contract
plt.subplot(2, 2, 1)
plt.title("Frecuencia de Pago de Servicio")
feature_used = "Contract"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# PaperlessBilling
plt.subplot(2, 2, 2)
plt.title("Factura Digital")
feature_used = "PaperlessBilling"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

# Contract
plt.subplot(2, 2, 3)
plt.title("Método de Pago de Servicio")
feature_used = "PaymentMethod"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')

plt.show()

# Histogramas charts
f, ax = plt.subplots(figsize=(8, 10))

# MonthlyCharges
plt.subplot(2, 1, 1)
plt.hist(dataset['MonthlyCharges'], color = 'red')
plt.title('Histograma de Cargo Mensual al Cliente')
plt.xlabel('Cargo Mensual')

plt.show()
# TotalCharges is of 'Object' datatype. Looking at the data set it should be of 'float' data type
# Convert 'TotalCharges' to numeric data type
# pd.to_numeric(dataset['TotalCharges'])
# ValueError: Unable to parse string " " at position 488.

dataset.iloc[488]
# The error is because of whitespace in the 'TotalCharges' column. If there is a missing observation pandas would have filled 
# with NaN but since there is a whitespace character the entire feature is converted to string data type.
dataset = dataset.replace(" ", 0)
# Ver si los datos estan completos o hay algún NaN
#convert to float type
# Convert 'TotalCharges' to numeric data type
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])
dataset.isnull().sum()
# Hacemos copia del dataset con las variables originales para análisis posteriores
dataset1 = dataset.copy()
dataset.info()
dataset.describe()
# Histogramas charts
f, ax = plt.subplots(figsize=(8, 10))

# TotalCharges
plt.subplot(2, 1, 1)
plt.hist(dataset['TotalCharges'], color = 'red')
plt.title('Histograma de Cargo Total al Cliente')
plt.xlabel('Cargo Total')

plt.show()
# Churn
f, ax = plt.subplots(figsize=(8, 10))

# SeniorCitizen
plt.subplot(2, 2, 1)
plt.title('Churn')
feature_used = "Churn"
sums = dataset[feature_used].value_counts()
plt.pie(sums.values, labels=sums.index, autopct='%1.1f%%')
plt.show()
#3. MODELADO DE ANN

