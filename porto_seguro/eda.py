#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: karlvandesman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 59)

#%% ---------------
### Reading the csv
### ---------------

df = pd.read_csv('data/train.csv', index_col='id')

print(df.head())
print()

columns = df.columns.values

columns_bin = [ i for i in range(len(columns)) if df.columns.values[i].endswith('bin')]

columns_cat = [ i for i in range(len(columns)) if df.columns.values[i].endswith('cat')]

#%% ------------------
### Splitting features
### ------------------

# Categorical, numerical and binary

features = df.columns.tolist()
features.remove('target')

numeric_features = [x for x in features if x[-3:] not in ['bin', 'cat']]
categorical_features = [x for x in features if x[-3:]=='cat']
binary_features = [x for x in features if x[-3:]=='bin']


#%% ----------------------
### Descriptive statistics
### ----------------------

# Let's see how imbalanced the dataset is
plt.bar(['Not filed', 'filed'], 100*df.target.value_counts()/df.shape[0],
        color=['r', 'b'])
plt.title('Distributions of claims')
plt.ylabel('%')
plt.show()

# Plotting the boxplot for each numeric feature
df[numeric_features].boxplot(figsize=(30,30))

print("Percentage of 1's by feature:", df[df.columns[columns_bin]].mean()*100)

#print('Statistics for binary features:\n', df_bin.describe())

print('Statistics for nominal features:\n', df[categorical_features].describe())

# Numerical data 
print('Statistics for features with interval values:\n', 
      df[numeric_features].describe())

#%% --------------
### Missing values
### --------------

# The missing values are represented as -1

#print('Number of missing values by variable (%)')
#(df==-1).sum() * 100/len(df)

vars_with_missing = []

print('Number of missing values by variable:')
for f in df.columns:
    missings = df[df[f] == -1][f].count()
    if missings > 0:
        vars_with_missing.append(f)
        missings_perc = missings/df.shape[0]
        
        print('{} has {} missing records ({:.4%})'.format(f, missings, missings_perc))
        
print('In total, there are {} of {} variables with missing values'.format(len(vars_with_missing), df.shape[1]))

# For each level of variable, there is a proper imputation methods

# Categorical -> substitute by its mode
# Numerical -> substitute by its mean/median (consider the outliers)

#%% ---------------
### Binary Features
### ---------------

# Distribution of binary features grouped by class

for column in (binary_features+categorical_features):
    ### Figure initiation 
    fig = plt.figure(figsize=(18,12))
    
    ### Number of occurrences per binary value - target pair
    ax = sns.countplot(x=column, hue="target", data=df, ax = plt.subplot(211));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Number of occurrences', fontsize=14)
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 1000,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 

#%% ---------------
### Categorical Features
### ---------------

# Distribution of categorical features grouped by class
# (The last categorical feature is omitted)
for column in categorical_features[:-1]:
    # Figure initiation
    fig = plt.figure(figsize=(18,12))
    
    ### Number of occurrences per categoty - target pair
    ax = sns.countplot(x=column, hue="target", data=df, ax = plt.subplot(211));
    # X-axis Label
    plt.xlabel(column, fontsize=14);
    # Y-axis Label
    plt.ylabel('Number of occurrences', fontsize=14)
    # Adding Super Title (One for a whole figure)
    plt.suptitle('Plots for '+column, fontsize=18);
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 1000,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14)         

