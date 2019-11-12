import pandas as pd

pd.set_option('display.max_colwidth', 40)

class Dataset:

    def __init__(self):
        self.train_dataset = \
        pd.read_csv('./data/segmentation.data', 
                    skiprows=2).drop(['REGION-PIXEL-COUNT'], axis=1)
        
        self.test_dataset = \
        pd.read_csv('./data/segmentation.test', 
                    skiprows=2).drop(['REGION-PIXEL-COUNT'], axis=1)
        
    def split_views(self, dataset_):
        col_shift = 9 - (19 - self.train_dataset.shape[1])
        
        return dataset_[:, 0:col_shift], dataset_[:, col_shift:];

    def getLabels(self, dataset_):
        return dataset_.index;

    def info(self, dataset_):
        print("Dimension of dataset:")
        print(dataset_.shape)
        print()

        print("Number of Unique Values by feature:")
        print(dataset_.nunique())
        print()

        print("Sample of data:")
        print(dataset_.sample());

        print("Descritive statistics:")
        print(dataset_.describe())