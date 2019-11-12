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
    cluster_centers : array, shape [n_clusters, n_features]
        Coordinates of cluster centers.

    sigma_term: float
        The quotient of the gaussian function, 2*sigma^2.

    membership_degree: array, shape [n_samples, n_clusters]
        Fuzzy partition membership degree for each sample, ui = (ui1, ..., uic)
        with values between 0 and 1, with sum over clusters equals to one.

    weights: array, shape [n_clusters, n_features]
        Vector of weights that parameterize the adaptative distances.

    labels : array, shape [n_samples,]
        Labels of each point

    References
    ----------
    [1] M.R.P. Ferreira, F.A.T. de Carvalho.
        "Kernel fuzzy c-means with automatic variable weighting",  Fuzzy Sets 
        and Systems, 237, 1-46, 2014

    [2] R. Winkler, F. Klawonn, R. Kruse
        "Fuzzy c-means in high dimensional spaces", Int. J. Fuzzy Syst. Appl. 1 
        (1) (2011) 1–16.

    [3] R. Winkler, F. Klawonn, R. Kruse
        "Problems of fuzzy c-means clustering and similar algorithms with high 
        dimensional data sets", Int. J. Fuzzy Syst. Appl. 1 (1) (2011) 1–16.
    """
    
    def __init__(self, n_clusters=7, m=1.6, max_iter=150, epsilon=10**(-10), 
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
        
        self.p = X.shape[1]
        
        # Calculating all the distances between 2 rows
        X_diff = np.linalg.norm(X[None, :, :] - X[:, None, :], axis=2)
        
        # X_diff = [ abs(X[None, :, i] - X[:, None, i]) for i in range(p) ]
        
        # 2*sigma^2: obtained by the quantiles mean's, for each feature
        sigma_term = [np.mean([np.quantile(X_diff[i], 0.1), 
                               np.quantile(X_diff[i], 0.9)]) for i in range(self.p)]
        
        self.sigma_term = sigma_term
    
    def __gaussianKernel(self, X, V):
        '''Implements the Gaussian kernel between two arrays.
        
        Parameters
        ----------
        X: array-like
        V: array-like
        sigma_term: float
            The quotient of the gaussian function, 2*sigma^2.
        '''
        
        return np.exp(-(np.linalg.norm(X-V, axis=2)**2/self.sigma_term))

    def __suitable_squared_dist(self, X, V):
      ''' Calculates the suitable squared distance between the pattern x_k and
      the cluster centroid v_i. This implementation consider the constraint 
      that the product of the weights of the variables for each cluster must
      be equal to one (Equation 22, [1]).
      '''
      
      phi_squared = [ self.weights[j] * 2 * (1 - self.__gaussianKernel(X[j], V[j])) 
                    for j in range(self.p) ]
      
      return np.sum(phi_squared)

    def predict(self, X):
        """Predict the class labels for the provided data.
        
        Parameters
        ----------
        X : array-like, shape (n_queries, n_features)
            
        Returns
        -------

        """
                
        n, p = X.shape
        
        np.random.seed(self.random_state)
        
        # Randomly initialize the fuzzy membership degree
        U = np.random.rand(n, self.n_clusters)
        U = U/np.sum(U)
        
        # By the condition of weight's product to be 1, all weights are 
        # initiallized as 1
        self.weights = np.ones((p, 1))

        V = np.empty((self.n_clusters, p))
        
        self.cost = 100 #just initializing a value
        iter_ = 0

        while((self.cost < self.epsilon) and (iter_ < self.max_iter)):
          iter_ += 1

          # TODO: Initial implementation was made using for loops, but it can 
          # be done using vectorization for a faster algorithm

          # *** Update the cluster centroids ***
          # Equation (27)
        	
          for i in range(self.n_clusters):
            for j in range(p):
              sum_num = 0
              sum_den = 0
              for k in range(n):
                sum_num += U[i, k]**self.m * self.__gaussianKernel(X[k, j], V[i, j]) * X[k, j]
                sum_den += U[i, k]**self.m * self.__gaussianKernel(X[k, j], V[i, j])
                  
                V[i, j] = sum_num/sum_den
          
          #** Update the weights ***
          # Equation (31)

          sum_num = np.zeros((p, 1))
          for j in range(p):
            for l in range(p):
              sum_num = 0
              sum_den = 0
              for i in range(self.n_clusters):
                for k in range(n):
                  sum_num[l] += U[i, k]**(self.m) * 2 * (1 - self.__gaussianKernel(X[k, l], V[i, l]))
                  sum_den += U[i, k]**(self.m) * 2 * (1 - self.__gaussianKernel(X[k, j], V[i, j]))
            self.weights[j] = (np.prod(sum_num))**p/sum_den
            
            # *** Membership degree update ***
            # Equation (32)
            for k in range(n):
              sum_terms = 0
              for i in range(self.n_clusters):
                num = self.__suitable_squared_dist(X[k, :], V[i, :])
                for h in range(self.n_clusters):
                  den = self.__suitable_squared_dist(X[k, :], V[i, :])
                  sum_terms += (num/den)**(1/(self.m-1))
                U[i, k] = 1/sum_terms
            
            self.membership_degree = U 

            # *** Cost function ***
            # Equation (15)
            sum_terms = 0
            for i in range(self.n_clusters):
              for k in range(n):
                sum_terms += U**self.m * self.__suitable_squared_dist(X[k, :], 
                                                                      V[i, :])
            self.cost = sum_terms
              