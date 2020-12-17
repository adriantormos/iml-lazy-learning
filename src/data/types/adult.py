import string
from src.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd


class AdultDataset(Dataset):

    def __init__(self, config, verbose):
        self.name = 'adult'

        train_datasets, test_datasets = self.load_k_fold_datasets()

        train_data = train_datasets[0]
        test_data = test_datasets[0]
        whole_data = train_data
        whole_data.append(test_data, ignore_index=True)
        self.one_hot = OneHotEncoder().fit(whole_data.select_dtypes(include=np.object).drop('class', axis=1))
        self.min_max = MinMaxScaler().fit(whole_data.select_dtypes(include=np.number))
        self.label = LabelEncoder().fit(whole_data['class'])

        super(AdultDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> string:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        # TODO implement this
        categorical_columns = data.select_dtypes(include=np.object).drop('class', axis=1)
        numeric_columns = data.select_dtypes(include=np.number)

        numeric_columns = self.min_max.transform(numeric_columns)
        categorical_columns = self.one_hot.transform(categorical_columns).toarray()
        values = np.array([np.append(numeric_columns[i], categorical_columns[i]) for i, _ in enumerate(numeric_columns)])

        labels = self.label.transform(data['class'])

        return values, labels
