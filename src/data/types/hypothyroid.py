from src.data.dataset import Dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class HypothyroidDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        self.name = 'hypothyroid'
        # self.balance = config['balance']  # Not implemented yet
        # self.only_numerical = config['only_numerical']
        # self.class_feature = 'Class'
        # self.numerical_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
        # self.null_values = [b'?']
        # self.classes_to_numerical = config['classes_to_numerical']
        # self.verbose = verbose
        # self.preprocessed_data = self.preprocess_dataset()

        train_datasets, test_datasets = self.load_k_fold_datasets()

        train_data = train_datasets[0]
        test_data = test_datasets[0]
        whole_data = train_data
        whole_data.append(test_data, ignore_index=True)
        self.one_hot = OneHotEncoder().fit(whole_data.select_dtypes(include=np.object).drop('Class', axis=1))
        self.min_max = MinMaxScaler().fit(whole_data.select_dtypes(include=np.number))
        self.label = LabelEncoder().fit(whole_data['Class'])

        super(HypothyroidDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> str:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        categorical_columns = data.select_dtypes(include=np.object).drop('Class', axis=1)
        numeric_columns = data.select_dtypes(include=np.number)

        numeric_columns = self.min_max.transform(numeric_columns)
        categorical_columns = self.one_hot.transform(categorical_columns).toarray()
        values = np.array([np.append(numeric_columns[i], categorical_columns[i]) for i, _ in enumerate(numeric_columns)])

        labels = self.label.transform(data['Class'])

        return values, labels

    # def __init__(self, config, verbose):
    #     super(HypothyroidDataset, self).__init__(config, verbose)
    #     # self.data, _ = load_arff('hypothyroid')
    #     # self.data = pd.DataFrame(self.data)
    #     self.balance = config['balance']  # Not implemented yet
    #     self.only_numerical = config['only_numerical']
    #     self.class_feature = 'Class'
    #     self.numerical_features = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    #     self.null_values = [b'?']
    #     self.nominal_features = [name for name in self.data.columns if name not in self.numerical_features + [self.class_feature]]
    #     self.classes_to_numerical = config['classes_to_numerical']
    #     self.verbose = verbose
    #
    #     train_datasets, test_datasets = self.load_k_fold_datasets()
    #
    #     train_data = train_datasets[0]
    #     test_data = test_datasets[0]
    #     whole_data = train_data
    #     whole_data.append(test_data, ignore_index=True)
    #     self.one_hot = OneHotEncoder().fit(whole_data.select_dtypes(include=np.object).drop('class', axis=1))
    #     self.min_max = MinMaxScaler().fit(whole_data.select_dtypes(include=np.number))
    #     self.label = LabelEncoder().fit(whole_data['class'])
    #
    #     self.preprocessed_data = self.preprocess_dataset()
    #
    # def get_raw_data(self) -> (np.ndarray, np.ndarray):
    #     if self.only_numerical:
    #         values = self.data[self.numerical_features].to_numpy()
    #     else:
    #         values = self.data.loc[:, self.data.columns != self.class_feature].to_numpy()
    #     labels = self.data[self.class_feature].to_numpy()
    #     return values, labels
    #
    # def get_raw_dataframe(self) -> pd.DataFrame:
    #     if self.only_numerical:
    #         data = self.data[self.numerical_features + [self.class_feature]]
    #     else:
    #         data = self.data
    #     return data
    #
    # def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
    #     if self.only_numerical:
    #         values = self.preprocessed_data[self.numerical_features].to_numpy()
    #     else:
    #         values = self.preprocessed_data.loc[:, self.preprocessed_data.columns != self.class_feature].to_numpy()
    #     labels = self.preprocessed_data[self.class_feature].to_numpy()
    #     return values, labels
    #
    # def get_preprocessed_dataframe(self) -> pd.DataFrame:
    #     if self.only_numerical:
    #         data = self.preprocessed_data[self.numerical_features + [self.class_feature]]
    #     else:
    #         data = self.preprocessed_data
    #     return data
    #
    # # Auxiliary methods
    #
    # def preprocess_dataset(self):
    #     data = self.data
    #
    #     if self.verbose:
    #         print('Started data preprocessing')
    #
    #     # Replace non orthodox nan values like ? for nan values
    #     for null_value in self.null_values:
    #         data.replace(null_value, None, inplace=True)
    #
    #     # Delete features with more than half of samples with NaN values
    #     if self.verbose:
    #         nan_count = data.isnull().sum().sum()
    #         print('    ', 'Total number of NaNs: ', nan_count, '; relative: ',
    #               (nan_count * 100) / (len(data.index) * len(data.columns)), '%')
    #
    #     columns_to_drop = []
    #     for feature_index in data.columns:
    #         nan_count = data[feature_index].isnull().sum()
    #         if nan_count > (len(data.index) / 2):
    #             columns_to_drop.append(feature_index)
    #     data.drop(columns=columns_to_drop, inplace=True)
    #     self.numerical_features = [name for name in self.numerical_features if name not in columns_to_drop]
    #     self.nominal_features = [name for name in self.nominal_features if name not in columns_to_drop]
    #     if self.verbose:
    #         print('    ', 'Deleted because of too many NaN values the features with name:', columns_to_drop)
    #
    #     # Numerical features -> replace the NaN values by the mean and normalize
    #     for feature_index in self.numerical_features:
    #         feature = data[feature_index]
    #
    #         # replace the NaN values by the mean
    #         nan_indexes = data.index[feature.isnull()].tolist()
    #         feature = feature.to_numpy()
    #         feature_without_nans = np.delete(feature, nan_indexes)
    #         mean = np.mean(feature_without_nans)
    #         feature[nan_indexes] = mean
    #
    #         # do normalization
    #         normalized_feature = min_max_normalize(feature)
    #         data[feature_index] = normalized_feature
    #
    #     # Nominal features -> replace the NaN values by the median
    #     if not self.only_numerical:
    #         for feature_index in self.nominal_features:
    #             feature = data[feature_index]
    #
    #             # replace the NaN values by the median
    #             nan_indexes = data.index[feature.isnull()].tolist()
    #             feature = feature.to_numpy()
    #             feature_without_nans = np.delete(feature, nan_indexes)
    #             unique, counts = np.unique(feature_without_nans, return_counts=True)
    #             median = unique[np.argmax(np.asarray(counts))]
    #             feature[nan_indexes] = median
    #             data[feature_index] = feature
    #
    #         # do hot encoding
    #         data = one_hot_encoding(data, self.nominal_features)
    #
    #     # Convert classes to numerical
    #     data[self.class_feature] = data[self.class_feature].str.decode("utf-8")
    #     data[self.class_feature] = data[self.class_feature].map(self.classes_to_numerical)
    #
    #     if self.verbose:
    #         print('Finished data preprocessing')
    #
    #     return data
