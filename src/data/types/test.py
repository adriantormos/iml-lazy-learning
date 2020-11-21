from src.data.dataset import Dataset
import numpy as np
import pandas as pd


class TestDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(TestDataset, self).__init__(config, verbose)
        self.values = np.asarray([[0.5, 1], [0, 0], [1, 0.5]])
        self.labels = np.asarray([0, 1, 1])

    def get_raw_data(self) -> (np.ndarray, np.ndarray):
        return self.get_preprocessed_data()

    def get_raw_dataframe(self) -> pd.DataFrame:
        return self.get_preprocessed_dataframe()

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        return self.values, self.labels

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        aux = np.concatenate((self.values, np.expand_dims(self.labels, axis=0).T), axis=1)
        return pd.DataFrame(data= aux,
                            index=[x for x in range(self.values.shape[0])],
                            columns=[x for x in range(self.values.shape[1] + 1)])
