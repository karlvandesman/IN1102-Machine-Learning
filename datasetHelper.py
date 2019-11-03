import pandas as pd

class Dataset:

    def __init__(self):
        self.train_dataset = pd.read_csv('./data/segmentation.data', skiprows=2).drop(['REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2'], axis=1);
        self.test_dataset = pd.read_csv('./data/segmentation.test', skiprows=2).drop(['REGION-PIXEL-COUNT', 'SHORT-LINE-DENSITY-5', 'SHORT-LINE-DENSITY-2'], axis=1);

    def split_views(self, dataset_):
        return  pd.DataFrame(data=dataset_).iloc[:,0:6].copy(), pd.DataFrame(data=dataset_).iloc[:,6:].copy();

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