#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__      = "Karl Sousa"
__email__       = "kvms@cin.ufpe.br"


import numpy as np
from timeit import default_timer as timer

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
        
        self.sigma_term = np.array(sigma_term)
    
    def __gaussianKernel(self, x, v, sigma_term):
        '''Implements the Gaussian kernel between two arrays.
        
        Parameters
        ----------
        X: array-like
        V: array-like
        sigma_term: float
            The quotient of the gaussian function, 2*sigma^2.
        '''
        
        return np.exp(-((x-v)**2/sigma_term))

    def __gaussianKernel2(self, X, V):
        '''Implements the Gaussian kernel between two arrays.
        
        Parameters
        ----------
        X: array-like
        V: array-like
        sigma_term: array-like
            The quotient of the gaussian function, 2*sigma^2.
        '''
        
        return np.exp(-(np.linalg.norm(X-V, axis=2)**2/self.sigma_term))


    def __suitable_squared_dist(self, X, V):
      ''' Calculates the suitable squared distance between the pattern x_k and
      the cluster centroid v_i. This implementation consider the constraint 
      that the product of the weights of the variables for each cluster must
      be equal to one (Equation 22, [1]).
      '''
      
      phi_squared = [ self.weights[j] * 2 * (1 - self.__gaussianKernel(X[j], V[j], self.sigma_term[j])) 
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
        print('Start running fuzzy clustering...')
        
        # Randomly initialize the fuzzy membership degree
        U = np.random.rand(n, self.n_clusters)
        U = U/np.sum(U, axis=1).reshape(-1, 1)  # normalizing such that
                                                # sum U[i] over c clusters = 1
        
        # By the condition of weight's product to be 1, all weights are 
        # initiallized as 1
        self.weights = np.ones((p, 1))
        
        V = np.empty((self.n_clusters, p))
        
        self.membership_degree = U
        #print('Initial value for U (membership degree)\n', U)
        
        self.cost = 100 #just initializing a value
        last_cost = 200
        
        iter_ = 0
        time = 0
        while((self.cost > self.epsilon) and (iter_ < self.max_iter) and abs((self.cost-last_cost)/last_cost)>0.001 ):
            start = timer()
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
                        term = U[k, i]**(self.m) * self.__gaussianKernel(X[k, j], V[i, j], self.sigma_term[j])
                        sum_den += term
                        sum_num += term * X[k, j]
                      
                    V[i, j] = sum_num/sum_den
            self.centroids = V
            
            #** Update the weights ***
            # Equation (31)
            sum_num = np.zeros((p, 1))
            for j in range(p):
                for l in range(p):
                    sum_num[l] = 0
                    sum_den = 0
                    for i in range(self.n_clusters):
                        for k in range(n):
                            sum_num[l] += U[k, i]**(self.m) * 2 * (1 - self.__gaussianKernel(X[k, l], V[i, l], self.sigma_term[l]))
                            sum_den += U[k, i]**(self.m) * 2 * (1 - self.__gaussianKernel(X[k, j], V[i, j], self.sigma_term[j]))
#                print('2*(1-Kernel):', 2*(1 - self.__gaussianKernel(X[k, l], V[i, l], self.sigma_term[l])))
#                print('For j=',j ,'sum_num:', sum_num)
#                print('for %d, weights_num_prod=%.12f, weights_den=%.12f' %(j, np.prod(sum_num), sum_den))

                self.weights[j] = (np.prod(sum_num**(1/p)))/(sum_den)

            # *** Membership degree update ***
            # Equation (32)
            for k in range(n):
                for i in range(self.n_clusters):
                    sum_terms = 0
                    num = self.__suitable_squared_dist(X[k, :], V[i, :])
                    for h in range(self.n_clusters):
                        den = self.__suitable_squared_dist(X[k, :], V[h, :])
                        sum_terms += (num/den)**(1/((self.m)-1))
                    U[k, i] = 1/sum_terms
            
            self.membership_degree = U

            # *** Cost function ***
            # Equation (15)
            sum_terms = 0
            for i in range(self.n_clusters):
                for k in range(n):
                    sum_terms += U[k, i]**self.m * self.__suitable_squared_dist(X[k, :], 
                                                                                V[i, :])
            last_cost = self.cost
            self.cost = sum_terms
            end = timer()
            time += end-start
            estimated_left = ((self.max_iter - iter_) * time/iter_)/60
            print('Iteration %d finished. Time elapsed: %.2fs | Total: %.2fs | Estimated time left: %.2fmin'%(iter_, end-start, time, estimated_left))
            print('Cost: ', self.cost)
        
    def info(self):
        """Present details about the fuzzy clustering
            
        Returns
        cost: float
            
        cluster_centers : array, shape [n_clusters, n_features]
            Coordinates of cluster centers.
    
        membership_degree: array, shape [n_samples, n_clusters]
            Fuzzy partition membership degree for each sample, ui = (ui1, ..., uic)
            with values between 0 and 1, with sum over clusters equals to one.
    
        weights: array, shape [n_clusters, n_features]
            Vector of weights that parameterize the adaptative distances.


        -------

        """
        
        print('Optimized cost:', self.cost)
        print()
        
        print('Prototypes (cluster centroids) %s:'%(self.centroids.shape,))
        print(self.centroids)
        print()
        
        print('Weights %s:'%(self.weights.shape,))
        print(self.weights)
        print()
        
        print("Membership degree %s:"%(self.membership_degree.shape,))
        print(self.membership_degree)
        print()
