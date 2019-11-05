import pandas as pd
import numpy as np
import warnings

from scipy.stats import multivariate_normal

# ignore warnings
warnings.filterwarnings('ignore')

class GaussianBayes:

    def fit(self, features, labels, classes):
        self.classes = classes;
        self.apriori_prob_classes =  self.__get_priori_prob(labels);
        self.mult_norm_dict = self.__mult_norm_by_class(features, labels);

    def predict(self, features):
        self.predicted_set = [];
       
        for row in features:
            dens_prob_classes = self.__calculate_dens_prob(row);
            probs = [( (self.apriori_prob_classes[class_] * dens_prob_classes[class_]) / sum(dens_prob_classes) ) for class_ in self.classes];
            
            self.predicted_set.append(probs);

    def __get_priori_prob(self, labels):
        num_classes = len(self.classes);
        apriori_prob = np.zeros(num_classes);
        
        for label in labels:
            apriori_prob[label] += 1;
        
        apriori_prob = apriori_prob / len(labels)
        
        return apriori_prob;

    def __mult_norm_by_class(self, features, labels):
        mult_norm_dict = [];  # list of dens_prob by each class
        classes = self.__group_by_class(features, labels);
    
        for class_ in self.classes:
            class_mean = np.mean(classes[str(class_)],axis=0);
            class_cov = np.cov(classes[str(class_)],rowvar=False);

            mult_norm_dict.append(multivariate_normal(class_mean, np.diag(class_cov)));
        
        return mult_norm_dict;

    def __group_by_class(self, features, labels):
        
        classes_dict = dict();
        
        for index, row in enumerate(features):
            curr_label = str(labels[index]);
              
            if curr_label not in classes_dict:
                classes_dict[curr_label] = [];
            
            classes_dict[curr_label].append(row);
        
        return classes_dict;

    def __calculate_dens_prob(self, row):
        return [ self.mult_norm_dict[class_].pdf(row) for class_ in self.classes];

