import pandas as pd
import numpy as np

from scipy.stats import multivariate_normal

class GaussianBayes:

    def fit(self, dataset, labels, classes):

        counts = labels[0].value_counts().to_dict();

        self.classes = classes;
        self.priori_prob_classes =  [ counts[class_]/ len(labels) for class_ in self.classes];
        self.mult_norm_dict = self.__summarize_by_class(dataset, labels);

    def fit_shape(self, features, labels):
        pass;

    def fit_RGB(self, features, labels):
        pass;

    def predict(self, dataset):
        self.predicted_set = [];
        for index, row in dataset.iterrows():
            dens_prob_classes = self.__calculate_dens_prob(row);
            probs = [( self.priori_prob_classes[class_] * dens_prob_classes[class_] / sum(dens_prob_classes) ) for class_ in self.classes];

            self.predicted_set.append(probs);

    def __summarize_by_class(self, dataset, labels):
        mult_norm_dict = [];  # list of dens_prob by each class
        gp_dataset = dataset.groupby(labels);
        
        for class_ in self.classes:
            class_dataset = gp_dataset.get_group(name=class_);
            class_mean = np.mean(class_dataset,axis=0);
            class_cov = np.cov(class_dataset,rowvar=False);

            mult_norm_dict.append(multivariate_normal(class_mean, np.diag(class_cov)));
    
        return mult_norm_dict;

    def __calculate_dens_prob(self, row):
        return [ self.mult_norm_dict[class_].pdf(row) for class_ in self.classes];

