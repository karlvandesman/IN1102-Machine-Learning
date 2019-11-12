import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import scipy.stats as st
from scipy.stats import wilcoxon

from models.fuzzy_clustering import FuzzyClustering
from models.bayesian_gaussian import GaussianBayes
from models.bayesian_knn import BayesianKNN
from datasetHelper import Dataset

def eda():
  
    dataset = Dataset()
    
    print(dataset.info)
    
    print('For the training data...')
    # Split Features and Classes
    dataset.info(dataset.train_dataset)

    print('For the testing data...')
    # Split Features and Classes
    dataset.info(dataset.test_dataset)

def bayesian():
    
    dataset = Dataset();

    # Split Features and Classes
#    X = dataset.train_dataset.values;
#    y = dataset.train_dataset.index;
    
    X = dataset.test_dataset.values;
    y = dataset.test_dataset.index;    
    
    classes, num_classes = np.unique(np.sort(y), return_counts=True);
    
    print('Dataset: Image Segmentation.')
    print('X: (%d, %d), y: (%d, 1)'%(X.shape[0], X.shape[1], 
          y.shape[0]))
    print()

    # Transform labels - categoric <->  numeric
    encoding = LabelEncoder();
    encoding.classes_ = classes;

    y = encoding.transform(y);
    classes = encoding.transform(classes);
    
    # Parameters
    L = 2		# quantity of views
    seed = 10
    max_k_neighbors = 20
    k = len(classes)
    
    # Cross-validation: "30 times ten-fold"
    n_folds = 10
    n_repeats = 30
    
    # Stardard scaler object
    scaling_data = StandardScaler()
    
    rkf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, 
                                  random_state=seed);
    
    accuracy_gaussian = []
    accuracy_KNN = []

    print('Running %d repeated %d-fold...'%(n_repeats, n_folds))

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
        
        bayesian_KNN_shape = BayesianKNN()
        bayesian_KNN_RGB = BayesianKNN()

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
    	"%.4f ± %.8f"%(alpha, np.mean(accuracy_KFold_gaussian), gaussian_CI))
    print()
    print("Bayesian kNN accuracy for 30 times 10-Fold (⍺=%.2f): \n"
    	"%.4f ± %.8f"%(alpha, np.mean(accuracy_KFold_KNN), KNN_CI))
    print()

    # Comparing the values generated by the RepeatedStratifiedKFold
    # The Wilcoxon test computes di = xi - yi
    # H0: accuracyGaussian > accuracyKNN
    # H1: accuracyGaussian < accuracyKNN
    statistic, pvalue = wilcoxon(accuracy_KFold_gaussian, accuracy_KFold_KNN)
    
    print('*** Wilcoxon signed-ranks test *** \n'
    	'p-value = %.8f' % pvalue)
    
    alpha = 0.05
    
    if(pvalue>alpha):
    	print('We have strong evidence (alpha = %.2f) that Gaussian classifier '
    	'performed better than the kNN classifier'%alpha)
    else:
    	print('We have strong evidence (alpha = %.2f) that kNN classifier '
    	'performed better than the Gaussian classifier'%alpha)


def fuzzy_clustering():

    dataset = Dataset();
 
    X = dataset.train_dataset.values
    
    scaling_data = StandardScaler()
    scaling_data.fit(X)
    X = scaling_data.transform(X)
    
    X_shape, X_RGB = dataset.split_views(X)
    
    fuzzy_clustering_shape = FuzzyClustering()
    fuzzy_clustering_RGB = FuzzyClustering()

    fuzzy_clustering_shape.fit(X_shape)
    fuzzy_clustering_RGB.fit(X_RGB)
    
    print("2*sigma_term^2: ", fuzzy_clustering_shape.sigma_term)
    print("2*sigma_term^2: ", fuzzy_clustering_RGB.sigma_term)
    
    fuzzy_clustering_shape.predict(X_shape)
    fuzzy_clustering_RGB.predict(X_RGB)

    print("Custo:", fuzzy_clustering_shape.cost)
    print("membership_degree: \n", fuzzy_clustering_shape.membership_degree)

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
    