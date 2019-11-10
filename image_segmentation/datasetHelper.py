import pandas as pd

class Dataset:

    def __init__(self):
        self.train_dataset = pd.read_csv('./data/segmentation.data', skiprows=2)
        self.test_dataset = pd.read_csv('./data/segmentation.test', skiprows=2)

    def split_views(self, dataset_):
        return dataset_[:,0:6], dataset_[:,6:];

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