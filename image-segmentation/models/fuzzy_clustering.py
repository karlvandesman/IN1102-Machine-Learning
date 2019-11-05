	#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"


import numpy as np

class FuzzyClustering:
    """ Class implementing the Variable-wise kernel fuzzy clustering algorithm 
    with kernelization of the metric.

    Parameters
    ----------
    n_clusters : integer (2 < n_clusters <= n_samples)
        Number of clusters to be assigned.

    m : float, optional (default=1.6)
        Fuzzification parameter. A sugestion given by R. Winkler, F. Klawonn 
        and R. Kruse, in papers about fuzzy c-means clustering, is 
        m = (2 + p)/p, where p is the number of variables, as a good choice for
        the fuzzification parameter.

    max_iter : integer, optional (default=150)
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations.

    epsilon: float, optional (default=10^(-10))
        Tolerance for the convergence of the adequacy criterion.

    random_state: integer or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    cluster_centers_ : array, shape [n_clusters, n_features]
        Coordinates of cluster centers.

    membership_degree_: array, shape [n_samples, n_clusters]
        Fuzzy partition membership degree for each sample, ui = (ui1, ..., uic)
        with values between 0 and 1, with sum over clusters equals to one.

    weights_: array, shape [n_clusters, n_features]
        Vector of weights that parameterize the adaptative distances.

    labels_ : array, shape [n_samples,]
        Labels of each point

    References
    ----------
    M.R.P. Ferreira, F.A.T. de Carvalho.
        "Kernel fuzzy c-means with automatic variable weighting",  Fuzzy Sets 
        and Systems, 237, 1-46, 2014

    R. Winkler, F. Klawonn, R. Kruse
        "Fuzzy c-means in high dimensional spaces", Int. J. Fuzzy Syst. Appl. 1 
        (1) (2011) 1–16.

    R. Winkler, F. Klawonn, R. Kruse
        "Problems of fuzzy c-means clustering and similar algorithms with high 
        dimensional data sets", Int. J. Fuzzy Syst. Appl. 1 (1) (2011) 1–16.
    """
    
    def __init__(self, n_clusters, m=1.6, max_iter=150, epsilon=10**(-10), 
                 random_state=None):
        
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.random_state = random_state
        
    def fit(self, X):
        """Fit the model using X as training data.
        
        Parameters
        ----------
        X : Training data. If array or matrix, shape [n_samples, n_features]
        
        Returns
        -------
        self : returns a "trained" model.
        """
        
        p = X.shape[1]
        
        # Calculating all the distances between 2 rows
        X_diff = np.linalg.norm(X[None, :, :] - X[:, None, :], axis=2)
        
        # X_diff = [ abs(X[None, :, i] - X[:, None, i]) for i in range(p) ]
        
        # 2*sigma^2: obtained by the quantiles mean's, for each feature
        sigma_term = [np.mean([np.quantile(X_diff[i], 0.1), 
                               np.quantile(X_diff[i], 0.9)]) for i in range(p)]
        
        self.sigma_term = sigma_term
        
        return self
    
    def __gaussianKernel(X, V, sigma_term):
        '''Implements the Gaussian kernel between two arrays.
        
        Parameters
        ----------
        X: array-like
        V: array-like
        sigma_term: float
            The quotient of the gaussian function, 2*sigma^2.
        '''
        
        return np.exp(-(np.linalg.norm(X-V, axis=2)**2/sigma_term))

    def predict(self, X):
        """Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            
        Returns
        -------
        y : array of shape [n_queries] or [n_queries, n_outputs]
            Class labels for each data sample.
        """
                
        n, p = X.shape
        
        np.random.seed(self.random_state)
        
        # Randomly initialize the fuzzy membership degree
        U = np.random.rand(n, self.n_clusters)
        U = U/np.sum(U)
        
        # By the condition of weight's product to be 1, all weights are 
        # initiallized as 1
        self.weights_ = np.ones((p, 1))

        V = np.empty(self.n_clusters, p)
        
        for _ in range(self.max_iter):
            # *** Update the cluster centroids ***
        	# Equation (27)
        	
            # TODO: Initial implementation was made using for loops, but it can 
            # be done using vectorization for a faster algorithm
            for i in range(self.n_clusters):
                for j in range(p):
                    sum_num = 0
                    sum_den = 0
                    for k in range(n):
                        sum_num += U[i, k]**self.m * self.__gaussianKernel(X[k, j], V[i, j]) * X[k, j]
                        sum_den += U[i, k]**self.m * self.__gaussianKernel(X[k, j], V[i, j])
                    
                    self.V[i, j] = sum_num/sum_den
                    #V[i, j] = np.sum( X[:, j])/np.sum(u**m * kernel, axis=1)
        
        	# *** Update the weights ***
        	# Equation (31)
        
        	# *** Membership degree update ***
        	# Equation (32)
