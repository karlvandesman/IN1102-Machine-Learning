	#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"


import numpy as np
from sklearn.neighbors import NearestNeighbors

class BayesianKNN:
    """ Class implementing the bayesian classifier based on k-Nearest Neighbors
    ----------
    n_neighbors : integer, optional (default = 3)
        Number of neighbors to use by default for :meth:`kneighbors` queries.

    Attributes
    ----------
    num_classes
        Number of classes
        
    References
    ----------
    Scikit-learn
        github.com/scikit-learn/scikit-learn/blob/master/sklearn/neighbors/
    """
    
    def __init__(self, n_neighbors=3):
        
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        
    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.
        
        Parameters
        ----------
        X : Training data. If array or matrix, shape [n_samples, n_features]
            
        y : Target values of shape = [n_samples] or [n_samples, n_outputs]
        
        Returns
        -------
        self : returns a "trained" model.
        """

        self.knn.fit(X)
        self.labels = y
        self.num_classes = len(np.unique(y))

        return self

    def predict_prob(self, X):
        """Return probability estimated for the test data X.

        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            Test samples.

        Returns
        -------
        prob : array of shape = [n_queries, n_classes], or a list of n_outputs
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        
        # Getting the neighbors index and distance for the test data X
        neighborsDist, neighborsIndex = self.knn.kneighbors(X)
                
        # Summing the neighbors of each class
        quantNeighbors = [ np.sum(self.labels[neighborsIndex]==j, axis=1) for j in range(self.num_classes) ]
        
        # Calculating the probability
        prob = np.array(quantNeighbors).T/self.n_neighbors
        
        return prob