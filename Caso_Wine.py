# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:43:23 2022

@author: ASUS
"""
#   Librerias
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans

# 1. ANALISIS EXPLORATORIO

# Cargar Datos
df = pd.read_csv("Caso_Wine.csv")
print(df.columns)
# Identificacion de variables
df.info()
df.describe()
df = df.drop(["Customer_Segment"], axis=1) #eliminamos la variable independiente
# Graficos de densidad
fig, ax = plt.subplots(5,3, figsize=(14,12))
axes_ = [axes_row for axes in ax for axes_row in axes]
for i,c in enumerate(df.columns):
    sns.distplot(df[c], ax = axes_[i])
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
# Variables con alta correlacion
corr = df.corr(method='spearman')
corr[corr > 0.6]
# Eliminar variables con alta correlacion
df = df.drop(["Flavanoids","Proanthocyanins","Color_Intensity","OD280","Proline"], axis=1)
# Escalando los datos
new_df= preprocessing.StandardScaler().fit_transform(df)
new_df = pd.DataFrame(new_df, columns=['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium','Total_Phenols','Nonflavanoid_Phenols','Hue'])
new_df.head()


# 2. MODELO DE CLUSTERING
# elbow method para obtener k
X = new_df[["Alcohol","Malic_Acid"]].values  
def elbow_method(epsilon, figure=False):
    wcss = [] 
    diff = np.inf
    i = 1
    
    while diff > epsilon:
        print(" K: {k}".format(k=i))
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300,n_init = 10, random_state = 0)
        kmeans.fit(X)
        
        if diff == np.inf:
            diff = kmeans.inertia_
        elif kmeans.inertia_ == 0:
            wcss.append(kmeans.inertia_)
            break
        else:
            diff = (wcss[-1] - kmeans.inertia_)/wcss[-1]
        wcss.append(kmeans.inertia_)
        i += 1
        
    if figure:
        plt.plot(range(0,len(wcss)), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Clusters Number')
        plt.ylabel('WCSS')
        plt.show()
    k = i-1
    return wcss, k

# Plot
epsilon = 0.05 
wcss, k = elbow_method(epsilon, figure=True)

# K-means
kmeans = KMeans(n_clusters = k, init= 'k-means++', max_iter = 300, n_init =10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

### Plot clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'blue',label = 'C1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'red',label = 'C2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green',label = 'C3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan',label = 'C4')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('WinesClusters')
plt.xlabel('X1: Alcohol')
plt.ylabel('X2: Malic_Acid')
plt.legend()
plt.show()

# 3. MODELO PREDICTIVO

# PCA



