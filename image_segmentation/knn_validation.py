#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 2019

@author: karlvandesman
"""

from datasetHelper import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.bayesian_knn import BayesianKNN

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

dataset = Dataset()

# Split Features and Classes
X = dataset.train_dataset.values
y = dataset.train_dataset.index

classes, num_classes = np.unique(np.sort(y), return_counts=True)

encoding = LabelEncoder()
encoding.classes_ = classes

y = encoding.transform(y)
classes = encoding.transform(classes)


#%%
seed = 10

y = y.reshape(-1, 1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, 
                                                  random_state=seed)

y_train = y_train.reshape(-1, )
y_val = y_val.reshape(-1, )

max_n_neighbors = 30
L = 2
k = len(classes)

scaling_data = StandardScaler()
scaling_data.fit(X_train)

X_train = scaling_data.transform(X_train)
X_val = scaling_data.transform(X_val)

#%%

accuracy_KNN = []

for i in range(1, max_n_neighbors+1): 
    X_train_shape, X_train_RGB = dataset.split_views(X_train)
    X_test_shape, X_test_RGB = dataset.split_views(X_val)
    
    bayesian_KNN_shape = BayesianKNN(n_neighbors=i)
    bayesian_KNN_RGB = BayesianKNN(n_neighbors=i)
    
    bayesian_KNN_shape.fit(X_train_shape, y_train)
    bayesian_KNN_RGB.fit(X_train_RGB, y_train)

    prob_KNN_shape = bayesian_KNN_shape.predict_prob(X_test_shape)
    prob_KNN_RGB = bayesian_KNN_RGB.predict_prob(X_test_RGB)    

    classes, num_classes = np.unique(np.sort(y_train), return_counts=True)
    PClasses = num_classes/len(y_train)

    # Using the shape and RGB views to combine the classifiers
    probClf2 = [ (1 - L) * PClasses[j] + prob_KNN_shape[:, j] 
                + prob_KNN_RGB[:, j] for j in range(k) ]
    
    # Fixing the shape
    probClf2 = np.vstack(probClf2).T
    
    # Get the class with highest probability
    y_pred_KNN = np.argmax(probClf2, axis=1)
    
    accuracy_KNN.append(accuracy_score(y_val, y_pred_KNN))

#%%

best_K = np.argmax(accuracy_KNN) + 1

print()
print('Best K: ', best_K)

x = range(1, max_n_neighbors+1)

plt.plot(x, accuracy_KNN, marker='o');
plt.xticks(np.arange(min(x), max(x), 3))
plt.title('Tuning kNN for the training/validation data')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
plt.grid()
plt.show()
