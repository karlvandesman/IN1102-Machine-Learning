#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karlvandesman
"""
import numpy as np
import pandas as pd

#%% ---------------
### Reading the csv
### ---------------

df = pd.read_csv('data/train.csv', index_col='id')

X = df.values[1:]
y = df.values[0]

#%% ---------------------
### Balancing the dataset
### ---------------------

seed = 595212

from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=seed)
X_resampled, y_resampled = cc.fit_resample(X, y)