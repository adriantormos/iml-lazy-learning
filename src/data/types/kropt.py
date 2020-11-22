import string
from src.data.dataset import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


class KroptDataset(Dataset):

    def __init__(self, config, verbose):
        self.name = 'kropt'
        self.balance = config['balance']
        self.verbose = verbose
        self.krops_category_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        super(KroptDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> string:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        columns = data.columns

        ss = MinMaxScaler()
        le = LabelEncoder()

        # Transform all columns to numeric values
        for col in columns:
            if col != 'game':
                data[col] = self.transform_krops_col_to_numeric(data[col], col)
        game_col = list(data['game'])
        data = data.drop(columns=['game'])
        data = ss.fit(data).transform(data)
        data = pd.DataFrame(data, columns=columns[:-1])
        data.insert(6, 'game', game_col)

        if self.balance:
            a = data[data['game'] == b'zero']
            data = data.append([data[data['game'] == b'zero']] * 75, ignore_index=True)
            data = data.append([data[data['game'] == b'one']] * 30, ignore_index=True)
            data = data.append([data[data['game'] == b'two']] * 10, ignore_index=True)
            data = data.append([data[data['game'] == b'three']] * 30, ignore_index=True)
            data = data.append([data[data['game'] == b'four']] * 12, ignore_index=True)
            data = data.append([data[data['game'] == b'five']] * 6, ignore_index=True)
            data = data.append([data[data['game'] == b'six']] * 6, ignore_index=True)
            data = data.append([data[data['game'] == b'seven']] * 5, ignore_index=True)
            data = data.append([data[data['game'] == b'eight']] * 2, ignore_index=True)
            data = data.append([data[data['game'] == b'nine']], ignore_index=True)
            data = data.append([data[data['game'] == b'ten']], ignore_index=True)
            data = data.append([data[data['game'] == b'fifteen']], ignore_index=True)
            data = data.append([data[data['game'] == b'sixteen']] * 8, ignore_index=True)
            print(data['game'].value_counts())

        le.fit(data['game'])
        data['game'] = le.transform(data['game'])

        values = data.loc[:, data.columns != 'game'].to_numpy()
        labels = data['game'].to_numpy()

        return values, labels

    # Auxiliary methods

    def transform_krops_col_to_numeric(self, column, column_name: str):
        if 'row' in column_name:
            return [int(x.decode('utf-8')) for x in column]
        return [self.krops_category_mapping[x.decode('utf-8')] for x in column]
