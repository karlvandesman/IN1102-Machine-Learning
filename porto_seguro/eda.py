#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karlvandesman
"""

import pandas as pd

pd.set_option('display.max_columns', 59)


#%% ---------------
### Reading the csv
### ---------------

df = pd.read_csv('data/train.csv')

print(df.head())
print()

columns = df.columns.values

columns_bin = [ i for i in range(len(columns)) if df.columns.values[i].endswith('bin')]

columns_cat = [ i for i in range(len(columns)) if df.columns.values[i].endswith('cat')]

#%% ---------
### Meta-data
### ---------

data = []
for f in df.columns:
    # Defining the role
    if f == 'target':
        role = 'target'
    elif f == 'id':
        role = 'id'
    else:
        role = 'input'
         
    # Defining the level
    if 'bin' in f or f == 'target':
        level = 'binary'
    elif 'cat' in f or f == 'id':
        level = 'nominal'
    elif df[f].dtype == float:
        level = 'interval'
    elif df[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'id':
        keep = False
    
    # Defining the data type 
    dtype = df[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(f_dict)
 
meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace=True)
 
#%% ----------------------
### Descriptive statistics
### ----------------------

# Let's see how unbalanced the dataset is
quantity_class = 100*df['target'].value_counts()/df.shape[0]

print('Quantity of values by class:')
print('0', quantity_class[0])
print('1', quantity_class[1])
print()


df_bin = df[meta[(meta['level']=='binary') & (meta.role =='input')].index]
print('Statistics for binary features:\n', df_bin.describe())

df_nominal = df[meta[(meta['level']=='nominal') & (meta.role=='input')].index]
print('Statistics for nominal features:\n', df_nominal.describe())

df_interval = df[meta[(meta['level']=='interval')].index]
print('Statistics for features with interval values:\n', df_interval.describe())

df_ordinal = df[meta[(meta['level']=='ordinal')].index]
print('Statistics for ordinal features:\n', df_ordinal.describe())

#%% --------------
### Missing values
### --------------

# The missing values are represented as -1

vars_with_missing = []

print('Number of missing values by variable:')
for f in df.columns:
    missings = df[df[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/df.shape[0]
        
        print('{} has {} records ({:.4%})'.format(f, missings, missings_perc))
        
print('In total, there are {} of {} variables with missing values'.format(len(vars_with_missing), df.shape[1]))

# For each level of variable, there is a proper imputation methods

# Categorical -> substitute by its mode
# Numerical -> substitute by its mean/median (consider the outliers)

#%% --------------
### Undersampling
### --------------

seed = 595212

X = df[meta[(meta.role=='input')].index].values
y = df.target.values

from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=seed)
X_resampled, y_resampled = cc.fit_resample(X, y)
