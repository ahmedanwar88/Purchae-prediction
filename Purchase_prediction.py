# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:58:17 2019

@author: Dell
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train_zero = []
index0 = []
y_train_one = []
index1 = []
for i in range(0,len(y_train)):
    if y_train[i] == 0:
        y_train_zero.append(y_train[i])
        index0.append(i)
    else:
        y_train_one.append(y_train[i])
        index1.append(i)

rows = np.size(X_test, axis = 0)
rows_train_one = np.size(index1, axis = 0)
rows_train_zero = np.size(index0, axis = 0)
X_train_one = []
X_train_zero = []
for i in index1:
    X_train_one.append(X_train[i])
for i in index0:
    X_train_zero.append(X_train[i])

conv_list_zero = []
summation = 0
for i in range(0,rows):
    for j in range(0, rows_train_zero):
        summation += (sum(np.convolve(X_test[i],X_train_zero[j])))
    conv_list_zero.append(summation)
    summation = 0

conv_list_one = []
summation = 0
for i in range(0,rows):
    for j in range(0, rows_train_one):
        summation += (sum(np.convolve(X_test[i],X_train_one[j])))
    conv_list_one.append(summation)
    summation = 0

dist_zero = []
dist_one = []
summation = 0
for i in range(0, rows):
    for j in range(0, rows_train_zero):
        summation += (np.sqrt(sum((X_test[i] - X_train_zero[j])**2)))
    dist_zero.append(summation)
    summation = 0

summation = 0
for i in range(0, rows):
    for j in range(0, rows_train_one):
        summation += (np.sqrt(sum((X_test[i] - X_train_one[j])**2)))
    dist_one.append(summation)
    summation = 0
    
theta0 = np.asarray(conv_list_zero)/np.asarray(dist_zero)
theta1 = np.asarray(conv_list_one)/np.asarray(dist_one)

predictions = []
for i in range(0, len(theta0)):
    if theta0[i] > theta1[i]:
        predictions.append(0)
    else:
        predictions.append(1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predictions)

Accuracy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]) * 100



