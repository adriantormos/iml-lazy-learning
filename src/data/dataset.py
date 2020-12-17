import os
import abc
import string
import pandas as pd
import numpy as np
from src.auxiliary.preprocessing_methods import shuffle_in_unison
from scipy.io.arff import loadarff


class DataLoader():

    def __init__(self, values, labels):
        self.values = values
        self.labels = labels
        self.iter = 0

    def next(self):
        if self.iter >= len(self.values):
            return None
        else:
            output_values = self.values[self.iter]
            output_labels = self.labels[self.iter]
            self.iter += 1
            return output_values, output_labels

    def get_index(self):
        return self.iter

    def reset(self):
        self.iter = 0

    def get_length(self):
        return len(self.values)


class Dataset(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'get_dataset_name') and
                callable(subclass.get_dataset_name) and
                hasattr(subclass, 'preprocess_data') and
                callable(subclass.preprocess_data) or
                NotImplemented)

    # Abstract class main methods

    def __init__(self, config, verbose):
        k_fold_train_datasets, k_fold_test_datasets = self.load_k_fold_datasets()
        train_values_preprocessed, train_labels_preprocessed, test_values_preprocessed, test_labels_preprocessed = self.preprocess_k_fold_datasets(k_fold_train_datasets, k_fold_test_datasets)
        self.train_loader = DataLoader(train_values_preprocessed, train_labels_preprocessed)
        self.test_loader = DataLoader(test_values_preprocessed, test_labels_preprocessed)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    # Abstract class auxiliary methods

    def load_k_fold_datasets(self) -> (list, list):
        dataset_name = self.get_dataset_name()
        train_datasets = []
        test_datasets = []
        directory = '../datasets/' + dataset_name
        filenames = [filename for filename in os.listdir(directory)]
        filenames.sort()
        for filename in filenames:
            if filename.endswith('train.arff'):
                data, _ = loadarff(os.path.join(directory, filename))
                train_datasets.append(pd.DataFrame(data))
            elif filename.endswith('test.arff'):
                data, _ = loadarff(os.path.join(directory, filename))
                test_datasets.append(pd.DataFrame(data))
        return train_datasets, test_datasets

    def preprocess_k_fold_datasets(self, train_datasets, test_datasets) -> (np.ndarray, np.ndarray):
        output_train_values = []
        output_train_labels = []
        output_test_values = []
        output_test_labels = []

        for index, train_dataset in enumerate(train_datasets):

            test_dataset = test_datasets[index]
            full_dataset = train_dataset.append(test_dataset, ignore_index=True)
            full_values, full_labels = self.preprocess_data(full_dataset)

            train_values = full_values[:len(train_dataset.index)]
            test_values = full_values[len(train_dataset.index):]
            train_labels = full_labels[:len(train_dataset.index)]
            test_labels = full_labels[len(train_dataset.index):]

            shuffle_in_unison(train_values, train_labels)
            shuffle_in_unison(test_values, test_labels)

            output_train_values.append(train_values)
            output_test_values.append(test_values)
            output_train_labels.append(train_labels)
            output_test_labels.append(test_labels)

        return output_train_values, output_train_labels, output_test_values, output_test_labels

    # Subclass main methods

    @abc.abstractmethod
    def get_dataset_name(self) -> string:
        raise NotImplementedError('Method not implemented in abstract class')

    @abc.abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        raise NotImplementedError('Method not implemented in abstract class')
