from src.data.dataset import Dataset
from src.auxiliary.preprocessing_methods import min_max_normalize, one_hot_encoding
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class SpliceDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        self.name = 'splice'
        self.class_feature = 'Class'
        self.null_values = [b'?']
        self.classes_to_numerical = {'N':0, 'IE':1, 'EI':2}
        self.nominal_features = None
        self.verbose = verbose
        super(SpliceDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> str:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):

        if self.nominal_features is None:
            self.nominal_features = [name for name in data.columns if
                                     name not in [self.class_feature]]

        # do hot encoding
        data = one_hot_encoding(data, self.nominal_features)

        # Convert classes to numerical
        data[self.class_feature] = data[self.class_feature].str.decode("utf-8")
        data[self.class_feature] = data[self.class_feature].map(self.classes_to_numerical)

        # Move class feature to the end
        cols_at_end = [self.class_feature]
        data = data[[c for c in data if c not in cols_at_end]
                + [c for c in cols_at_end if c in data]]

        values = data.loc[:, data.columns != self.class_feature].to_numpy()
        labels = data[self.class_feature].to_numpy()

        return values, labels
