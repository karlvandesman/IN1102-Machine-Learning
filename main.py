import argparse
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score

from models.gaussian import GaussianBayes
from models.fuzzy import Fuzzy
from datasetHelper import Dataset

def main(args):
    
    if args.command == 'gaussian':
       gaussian();

    elif args.command == 'knn':
        knn();

    elif args.command == 'fuzzy':
        fuzzy();

    else:
        gaussian();
        knn();
        fuzzy();


def gaussian():
    
    dataset = Dataset();

    # Split Features and Classes
    X = dataset.train_dataset.values;
    y = dataset.train_dataset.index;

    classes = np.unique(np.sort(y));

    # Transform labels - categoric <->  numeric
    encoding = LabelEncoder();
    encoding.classes_ = classes;

    y = encoding.transform(y);
    classes = encoding.transform(classes);
        
    #a) Cross-validation: "30 times ten-fold"
    L = 2	# quantity of views
    n_folds = 10
    n_repeats = 30
    seed = 10

    rkf = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed);

    # o vetor acuracia vai reunir os 300 valores obtidos pelo RepeatedKFold
    acuracy = [];

    for train_index, test_index in rkf.split(X):
        X_train, X_val = X[train_index], X[test_index];
        y_train, y_val = y[train_index], y[test_index];
    
        # Adicionar parte de treinamento/predição do classificador  
        print('training...')
        gb = GaussianBayes();
                
        shapeGb.fit(X_train_shape, y_train_shape, classes);
        RGBGb.fit(X_train_RGB, y_train_RGB, classes);

        print('testing...')
        shapeGb.predict(X_val_shape);
        RGBGb.predict(X_val_RGB);
        probClf1 = [];

        for i in range(len(X_val)):
            probClf1.append([(1 - L) * PClasse[j] + shapeGb.predicted_set[i][j] + RGBGb.predicted_set[i][j] for j in range(k) ]);
                
        # Agora deve-se obter o índice (qual classe) retornou o maior valor de probabilidade
        y_pred = np.argmax(probClf1, axis=1);
       
        print(np.vstack(y_pred))
        
        #acuracy.append(accuracy_score(y_val, y_pred));

    # Agora obtemos as médias de acurácias de 10 em 10 rodadas
    acuraciaKfold = np.asarray([ acuracy[i:i+10].mean() for i in range(0, n_folds*n_repeats, 10) ]);

def knn():
    pass;

def fuzzy():
    fz = Fuzzy();
    fz.__str__();

if __name__ == '__main__':
    parser = argparse.ArgumentParser();
    parser.add_argument('-c', '--command', type=str, help='enter a command [gaussian, knn ,fuzzy] to run the specific model. Default is [all]', action='store', default='all');
    args = parser.parse_args();

    main(args);
    