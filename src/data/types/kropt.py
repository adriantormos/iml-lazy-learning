import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from src.data.dataset import Dataset


class KroptDataset(Dataset):

    def __init__(self, config, verbose):
        self.name = 'kropt'
        self.balance = config['balance']
        self.encoding = config['encoding']
        self.verbose = verbose
        self.krops_category_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
        super(KroptDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> string:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):

        def _transform_krops_col_to_numeric(column, column_name: str):
            if 'row' in column_name:
                return [int(x.decode('utf-8')) for x in column]
            return [self.krops_category_mapping[x.decode('utf-8')] for x in column]

        def _numeric_encode(data_to_encode: pd.DataFrame) -> pd.DataFrame:
            ss = MinMaxScaler()
            for col in columns:
                if col != 'game':
                    data_to_encode[col] = _transform_krops_col_to_numeric(data_to_encode[col], col)
            game_col = list(data_to_encode['game'])
            data_to_encode = data_to_encode.drop(columns=['game'])
            data_to_encode = ss.fit(data_to_encode).transform(data_to_encode)
            data_to_encode = pd.DataFrame(data_to_encode, columns=columns[:-1])
            data_to_encode.insert(6, 'game', game_col)
            return data_to_encode

        def _one_hot_encode(data_to_encode: pd.DataFrame) -> pd.DataFrame:
            oh = OneHotEncoder()
            # Drop label column
            game_column = data_to_encode['game']
            data_to_encode = data_to_encode.drop('game', axis=1)
            # Apply one-hot encoding to data
            oh = oh.fit(data_to_encode)
            data_to_encode = oh.transform(data_to_encode).toarray()
            data_to_encode = pd.DataFrame(data=data_to_encode)
            data_to_encode['game'] = game_column
            return data_to_encode

        def _balance(to_balance: pd.DataFrame) -> pd.DataFrame:
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'zero']] * 75, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'one']] * 30, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'two']] * 10, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'three']] * 30, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'four']] * 12, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'five']] * 6, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'six']] * 6, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'seven']] * 5, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'eight']] * 2, ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'nine']], ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'ten']], ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'fifteen']], ignore_index=True)
            to_balance = to_balance.append([to_balance[to_balance['game'] == b'sixteen']] * 8, ignore_index=True)
            return to_balance

        columns = data.columns

        # Data encoding
        if self.encoding == 'numeric':
            data = _numeric_encode(data)
        elif self.encoding == 'one_hot_encoding':
            data = _one_hot_encode(data)
        else:
            raise NotImplementedError

        # Dataset balancing
        if self.balance:
            data = _balance(data)

        le = LabelEncoder()
        le = le.fit(data['game'])
        data['game'] = le.transform(data['game'])

        values = data.loc[:, data.columns != 'game'].to_numpy()
        labels = data['game'].to_numpy()

        return values, labels

    # Auxiliary methods
