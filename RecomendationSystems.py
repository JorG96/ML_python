# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 22:47:42 2022

@author: ASUS
"""
# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.utils.data

# Creating RBM class
class RBM():
    def __init__(self, nv, nh, batch_size, nb_epoch, k_steps, learning_rate, verbose):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        self.nh = nh
        self.nv = nv
        self.verbose = verbose
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.k_steps = k_steps
        self.learning_rate = learning_rate
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def update_weights(self, v0, vk, ph0, phk):
        learning_rate = self.learning_rate
        self.W += learning_rate*(torch.t(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)))
        self.b += learning_rate*(torch.sum((v0 - vk), 0))
        self.a += learning_rate*(torch.sum((ph0 - phk), 0))
        
    def train(self, training_set):
        batch_size = self.batch_size
        nb_epoch = self.nb_epoch
        k_steps = self.k_steps
        verbose = self.verbose
        
        # Training the RBM
        for epoch in range(1, nb_epoch + 1):
            train_loss = 0
            s = 0.
            nb_users = len(training_set)
            for id_user in range(0, nb_users - batch_size, batch_size):
                vk = training_set[id_user:id_user+batch_size]
                v0 = training_set[id_user:id_user+batch_size]
                ph0,_ = self.sample_h(v0)
                for k in range(k_steps):
                    _,hk = self.sample_h(vk)
                    _,vk = self.sample_v(hk)
                    vk[v0<0] = v0[v0<0]
                phk,_ = self.sample_h(vk)
                self.update_weights(v0, vk, ph0, phk)
                train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                s += 1.
            if verbose:
                print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
            
    def evaluate(self, test_set):
        verbose = self.verbose
        nb_users = len(test_set)
        
        # Testing the RBM
        test_loss = 0
        s = 0.
        for id_user in range(nb_users):
            v = training_set[id_user:id_user+1]
            vt = test_set[id_user:id_user+1]
            if len(vt[vt>=0]) > 0:
                _,h = self.sample_h(v)
                _,v = self.sample_v(h)
                test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
                s += 1.
        if verbose:  
            print('test loss: '+str(test_loss/s))
        return test_loss/s
    
    def predict(self, v_user):
        _,h = self.sample_h(v_user)
        _,v = self.sample_v(h)
        return v

# Load data
df_books = pd.read_csv('./Training/books.csv')
df_books = df_books[df_books['id']<=1000]
df_ratings = pd.read_csv('./Training/ratings.csv')
df_ratings = df_ratings[df_ratings['book_id'].isin(df_books['id'])][['user_id', 'book_id', 'rating']]
print(df_ratings.head())

# Pivot table
df_input = pd.pivot_table(df_ratings, values='rating', index=['user_id'],
                          columns=['book_id'], aggfunc=np.sum)
# Deal with NaN
df_input = df_input.fillna(-1)
print(df_input.head())

# Train/Test split
X = df_input.sample(frac=1).round() # Round ratings
len_train = int(np.round(0.8*len(X)))
X_train = X[:len_train]
X_test = X[len_train:]

# Converting to PyTorch tensors
training_set = X_train.values
test_set = X_test.values
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Did not liked)
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# TRAINING THE RBM

# Hyperparameters
nv = len(training_set[0])
nh = 100
batch_size = 100
nb_epoch = 10
k_steps = 10
learning_rate = 0.1
verbose = True

# Train model
rbm = RBM(nv, nh, batch_size, nb_epoch, k_steps, learning_rate, verbose)
rbm.train(training_set)

#EVALUATION
 
rbm.evaluate(test_set)

# Obtain an individual prediction
v_test = test_set[20:25]
v_pred = rbm.predict(v_test)

# Number of recomendations
n_recom = sum(v_pred.tolist()[0])

# Check book for that prediction
id_books = ([v_pred==1][0][0]) & ([v_test==-1][0][0])
id_books = id_books.tolist()

# Combine with book titles and see results for units with missing values
df_recom = pd.DataFrame({'id_check':id_books})
df_recom = df_recom.reset_index().rename(columns={'index':'book_id'}).merge(df_books)
print("Recommended books: ")
print(df_recom[df_recom['id_check']==True]['title'].head(10))

list_recom_books = []
for i in range(len(test_set)):
    v_pred = rbm.predict(test_set[i:i+1])
    n_recom = sum(v_pred.tolist()[0])
    list_recom_books.append(n_recom)
    
    # Check book for that prediction
    id_books = ([v_pred==1][0][0]) & ([v_test==-1][0][0])
    id_books = id_books.tolist()
    
print("Median of recommended books: ", np.median(list_recom_books))