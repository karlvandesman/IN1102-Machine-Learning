import numpy as np
import warnings

from scipy.stats import multivariate_normal

# ignore warnings
warnings.filterwarnings('ignore')

class GaussianBayes:

    def fit(self, X, y, classes):
        self.classes = classes;
        self.apriori_prob_classes =  self.__get_priori_prob(y)
        self.mult_norm_dict = self.__mult_norm_by_class(X, y)

    def predict_prob(self, X):
        
        prob = []
       
        for row in X:
            dens_prob_classes = self.__calculate_dens_prob(row)
                        
            probs = [( (self.apriori_prob_classes[class_]
                    * dens_prob_classes[class_]) / sum(dens_prob_classes))
                    for class_ in self.classes];
            
            prob.append(probs)
            
        return np.array(prob)
    
    def __get_priori_prob(self, y):
        num_classes = len(self.classes);
        apriori_prob = np.zeros(num_classes)
        
        for label in y:
            apriori_prob[label] += 1;
        
        apriori_prob = apriori_prob / len(y)
        
        return apriori_prob;

    def __mult_norm_by_class(self, X, y):
        '''
        Implements the multivariate normal distribution. 
        
        The parameters are the covariance matrix and the mean values from a 
        input X. These parameters can be simplified, where the covariance
        between features are zero, so the covariance matrix will be a diagonal
        matrix with different variances for each feature, and also 
        considering the same variance for all features. Similarly, the mean
        can be obtained for each feature, and also a general mean.
        '''
        
        mult_norm_dict = [];  # list of dens_prob by each class
        classes = self.__group_by_class(X, y);
        
        k = len(self.classes)
        
        for class_ in range (k):
            # Calculate the mean for the multivariate normal
            class_mean = np.mean(classes[str(class_)], axis=0)
                        
            # Calculate the variance for the multivariate normal
            class_var = np.var(classes[str(class_)])

            # Create the multivariate normal object for the given parameters
            mult_norm_dict.append(multivariate_normal(class_mean, class_var))
            
        return mult_norm_dict

    def __group_by_class(self, X, labels):
        
        classes_dict = dict()
        
        for index, row in enumerate(X):
            curr_label = str(labels[index]);
              
            if curr_label not in classes_dict:
                classes_dict[curr_label] = []
            
            classes_dict[curr_label].append(row)
        
        return classes_dict;

    def __calculate_dens_prob(self, row):
        return [ self.mult_norm_dict[class_].pdf(row)
                for class_ in self.classes]
