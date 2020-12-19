from src.data.dataset import Dataset
from src.auxiliary.preprocessing_methods import min_max_normalize, one_hot_encoding
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class SickDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        self.name = 'sick'
        self.class_feature = 'Class'
        self.numerical_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
        self.null_values = [b'?']
        self.classes_to_numerical = {"negative": 0, "sick": 1}
        self.nominal_features = None
        self.verbose = verbose
        super(SickDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> str:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):

        if self.nominal_features is None:
            self.nominal_features = [name for name in data.columns if
                                     name not in self.numerical_features + [self.class_feature]]

        # Replace non orthodox nan values like ? for nan values
        for null_value in self.null_values:
            data.replace(null_value, None, inplace=True)

        # Delete features with more than half of samples with NaN values
        if self.verbose:
            nan_count = data.isnull().sum().sum()
            print('    ', 'Total number of NaNs: ', nan_count, '; relative: ',
                  (nan_count * 100) / (len(data.index) * len(data.columns)), '%')

        columns_to_drop = []
        for feature_index in data.columns:
            nan_count = data[feature_index].isnull().sum()
            if nan_count > (len(data.index) / 2):
                columns_to_drop.append(feature_index)
        data.drop(columns=columns_to_drop, inplace=True)
        self.numerical_features = [name for name in self.numerical_features if name not in columns_to_drop]
        self.nominal_features = [name for name in self.nominal_features if name not in columns_to_drop]
        if self.verbose:
            print('    ', 'Deleted because of too many NaN values the features with name:', columns_to_drop)

        # Numerical features -> replace the NaN values by the mean and normalize
        for feature_index in self.numerical_features:
            feature = data[feature_index]

            # replace the NaN values by the mean
            nan_indexes = data.index[feature.isnull()].tolist()
            feature = feature.to_numpy()
            feature_without_nans = np.delete(feature, nan_indexes)
            mean = np.mean(feature_without_nans)
            feature[nan_indexes] = mean

            # do normalization
            normalized_feature = min_max_normalize(feature)
            data[feature_index] = normalized_feature

        # Nominal features -> replace the NaN values by the median
        for feature_index in self.nominal_features:
            feature = data[feature_index]

            # replace the NaN values by the median
            nan_indexes = data.index[feature.isnull()].tolist()
            feature = feature.to_numpy()
            feature_without_nans = np.delete(feature, nan_indexes)
            unique, counts = np.unique(feature_without_nans, return_counts=True)
            median = unique[np.argmax(np.asarray(counts))]
            feature[nan_indexes] = median
            data[feature_index] = feature

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
