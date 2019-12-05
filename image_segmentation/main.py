#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import scipy.stats as st
from scipy.stats import wilcoxon

from models.fuzzy_clustering_opt import FuzzyClustering
from models.bayesian_gaussian import GaussianBayes
from models.bayesian_knn import BayesianKNN
from datasetHelper import Dataset

from sklearn.metrics import adjusted_rand_score

# =============================================================================
# Exploratory Data Analysis
# =============================================================================
def eda():
  
    dataset = Dataset()
    
    print(dataset.info)
    
    print('For the training data...')
    # Split Features and Classes
    dataset.info(dataset.train_dataset)

    print('For the testing data...')
    # Split Features and Classes
    dataset.info(dataset.test_dataset)

# =============================================================================
# Bayesian classification (Gaussian and KNN)
# =============================================================================
def bayesian():
    
    dataset = Dataset()

    # Split Features and Classes
#    X = dataset.train_dataset.values;
#    y = dataset.train_dataset.index;
    
    X = dataset.test_dataset.values
    y = dataset.test_dataset.index 

    classes, num_classes = np.unique(np.sort(y), return_counts=True)
    
    print('Dataset: Image Segmentation.')
    print('X: (%d, %d), y: (%d, 1)'%(X.shape[0], X.shape[1], 
          y.shape[0]))

    #X_shape, X_RGB = dataset.split_views(X)
    #print('Shape (# of attributes): ', X_shape.shape[1])
    #print('RGB (# of attributes): ', X_RGB.shape[1])
    #print()

    # Transform labels - categoric <->  numeric
    encoding = LabelEncoder()
    encoding.classes_ = classes

    y = encoding.transform(y)
    classes = encoding.transform(classes)
    
    # Parameters
    L = 2		# number of views
    seed = 10
    k = len(classes)
    best_k_train = 14 # best n_neighbors for kNN (validation with training data)
    best_k_test = 9 # best n_neighbors for kNN (validation with test data)
    
    # Cross-validation: "30 times ten-fold"
    n_folds = 10
    n_repeats = 30
    
    # Stardard scaler object
    scaling_data = StandardScaler()
    
    rkf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, 
                                  random_state=seed)
    
    accuracy_gaussian = []
    accuracy_KNN = []

    print('Running %d repeated K-fold with K=%d...'%(n_repeats, n_folds))
    run_kfold = 0
    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standard scaler to normalize the data
        scaling_data.fit(X_train)

        X_train = scaling_data.transform(X_train)
        X_test = scaling_data.transform(X_test)
        
        X_train_shape, X_train_RGB = dataset.split_views(X_train)
        X_test_shape, X_test_RGB = dataset.split_views(X_test)

        classes, num_classes = np.unique(np.sort(y_train), return_counts=True)
        PClasses = num_classes/len(y_train)
        
        bayesian_gaussian_shape = GaussianBayes()
        bayesian_gaussian_RGB = GaussianBayes()
        
        bayesian_KNN_shape = BayesianKNN(n_neighbors=best_k_train)
        bayesian_KNN_RGB = BayesianKNN(n_neighbors=best_k_train)

        bayesian_gaussian_shape.fit(X_train_shape, y_train, classes)
        bayesian_gaussian_RGB.fit(X_train_RGB, y_train, classes)

        bayesian_KNN_shape.fit(X_train_shape, y_train)
        bayesian_KNN_RGB.fit(X_train_RGB, y_train)
                
        prob_gaussian_shape = bayesian_gaussian_shape.predict_prob(X_test_shape)
        prob_gaussian_RGB = bayesian_gaussian_RGB.predict_prob(X_test_RGB)

        prob_KNN_shape = bayesian_KNN_shape.predict_prob(X_test_shape)
        prob_KNN_RGB = bayesian_KNN_RGB.predict_prob(X_test_RGB)
        
        # Using the shape and RGB views to combine the classifiers
        probClf1 = [(1 - L) * PClasses[j] + prob_gaussian_shape[:, j]
                    + prob_gaussian_RGB[:, j] for j in range(k) ]
        
        probClf2 = [ (1 - L) * PClasses[j] + prob_KNN_shape[:, j] 
                    + prob_KNN_RGB[:, j] for j in range(k) ]
        
        # Fixing the shape
        probClf1 = np.vstack(probClf1).T 
        probClf2 = np.vstack(probClf2).T
        
        # Get the class with highest probability
        y_pred_gaussian = np.argmax(probClf1, axis=1)
        y_pred_KNN = np.argmax(probClf2, axis=1)
        
        accuracy_gaussian.append(accuracy_score(y_test, y_pred_gaussian))
        accuracy_KNN.append(accuracy_score(y_test, y_pred_KNN))
        
        run_kfold += 1
        if run_kfold%10==0: print('%dº 10-fold finished'%(run_kfold//10))
        
    print('Finished.\n')
    
    # Now we get the mean for the repeated K-folds
    accuracy_KFold_gaussian = \
                np.asarray([ np.mean(accuracy_gaussian[i:i+n_folds])
                            for i in range(0, n_folds*n_repeats, n_folds) ])
    
    accuracy_KFold_KNN = \
                np.asarray([ np.mean(accuracy_KNN[i:i+n_folds])
                            for i in range(0, n_folds*n_repeats, n_folds) ])

#    print("KFold accuracy bayesian gaussian:\n", accuracy_KFold_gaussian)
#    print()
#    print("KFold accuracy kNN bayesian:\n", accuracy_KFold_KNN)
#    print()
    
    # Calculates the point estimate for accuracy and its confidence interval
    alpha = 0.05
    conf = 1-alpha/2

    gaussian_CI = st.norm.ppf(conf) * np.std(accuracy_KFold_gaussian)/n_repeats
    KNN_CI = st.norm.ppf(conf) * np.std(accuracy_KFold_KNN)/n_repeats

    print("Bayesian gaussian accuracy for 30 times 10-Fold (⍺=%.2f): \n"
    	"%.6f ± %.6f"%(alpha, np.mean(accuracy_KFold_gaussian), gaussian_CI))
    print()
    print("Bayesian kNN accuracy for 30 times 10-Fold (⍺=%.2f): \n"
    	"%.6f ± %.6f"%(alpha, np.mean(accuracy_KFold_KNN), KNN_CI))
    print()

    # Comparing the values generated by the RepeatedStratifiedKFold
    # The Wilcoxon test computes di = xi - yi
    # H0: Same distribution
    # H1: Different distributions
    statistic, pvalue = wilcoxon(accuracy_KFold_gaussian, accuracy_KFold_KNN)
    
    print('*** Wilcoxon signed-ranks test *** \n'
    	'statistics = %.10f\np-value = %.8f' %(statistic, pvalue))
    
    alpha = 0.05
    
    if(pvalue>alpha):
    	print('We have strong evidence (alpha = %.2f) that Gaussian classifier '
    	'performed better than the kNN classifier'%alpha)
    else:
    	print('We have strong evidence (alpha = %.2f) that kNN classifier '
    	'performed better than the Gaussian classifier'%alpha)

# =============================================================================
#   Fuzzy Clustering
# =============================================================================
def fuzzy_clustering():

    dataset = Dataset()
 
    X = dataset.test_dataset.values
    y = dataset.test_dataset.index 

    classes, num_classes = np.unique(np.sort(y), return_counts=True)
    
    print('Dataset: Image Segmentation.')
    print('X: (%d, %d), y: (%d, 1)'%(X.shape[0], X.shape[1], 
          y.shape[0]))
    
    # Transform labels - categoric <->  numeric
    encoding = LabelEncoder()
    encoding.classes_ = classes

    y = encoding.transform(y)

    scaling_data = StandardScaler()
    scaling_data.fit(X)
    X = scaling_data.transform(X)
    
    X_shape, X_RGB = dataset.split_views(X)
    print('Shape (# of attributes): ', X_shape.shape[1])
    print('RGB (# of attributes): ', X_RGB.shape[1])
    print()

    seed_rgb = 3
    seed_shape = 100
    fuzzy_shape = FuzzyClustering(random_state=seed_shape)
    fuzzy_RGB = FuzzyClustering(random_state=seed_rgb)
    
    print('Fitting...')
    fuzzy_shape.fit(X_shape)
    fuzzy_RGB.fit(X_RGB)
    
    print('Calculated parameters:')
    print("2*sigma_term^2 (shape): ", fuzzy_shape.sigma_term)
    print("2*sigma_term^2 (RGB): ", fuzzy_RGB.sigma_term)
    print()

    print('Start running for shape view (seed=%d)...'%seed_shape)
    fuzzy_shape.predict(X_shape)
    fuzzy_shape.infos(view_name='shape', y_true=y)

    print('Start running for RGB view (seed=%d)...'%seed_rgb)
    fuzzy_RGB.predict(X_RGB)    
    fuzzy_RGB.infos(view_name='rgb', y_true=y)
    
    # Adjusted Rand index
    rand_views = adjusted_rand_score(fuzzy_shape.crisp_cluster, 
                                     fuzzy_RGB.crisp_cluster)
    
    print('Presenting adjusted Rand Index:')
    print('y true and shape: ', fuzzy_shape.rand_index)
    print('y true and RGB: ', fuzzy_RGB.rand_index)
    print('Between views: ', rand_views)        
    
def main(args):
    
    if args.command == 'bayesian':
        bayesian();

    elif args.command == 'fuzzy':
        fuzzy_clustering();

    elif args.command == 'eda':
        eda()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('-c', '--command', type=str, 
                        help=('enter a command [bayesian, fuzzy or eda] to run' 
                        'the specific model. Default is [eda]'), 
                        action='store', default='eda');
    args = parser.parse_args();

    main(args);
    