import abc
from time import time
from typing import Tuple
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.data.dataset import DataLoader


class SupervisedAlgorithm:

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'test') and
                callable(subclass.test) or
                NotImplemented)

    # Abstract class main methods

    def __init__(self, config, output_path, verbose):
        self.verbose = verbose

    def classify(self, train_loader: DataLoader, test_loader: DataLoader):
        self.overall_output_scores = np.zeros(train_loader.get_length())
        self.balanced_output_scores = np.zeros(train_loader.get_length())
        self.confusion_matrices = []

        self.total_train_time = 0
        self.total_test_time = 0

        print('Starting', train_loader.get_length(), '- fold')
        train_loader.reset()
        test_loader.reset()

        train_data = train_loader.next()
        test_data = test_loader.next()

        index = 0
        while train_data is not None:
            print('Doing step', index)
            init = time()
            train_values, train_labels = train_data
            test_values, test_labels = test_data

            init_train_time = time()
            self.train(train_values, train_labels)
            self.total_train_time += time() - init_train_time
            init_test_time = time()
            predicted_labels = self.test(test_values)
            self.total_test_time += time() - init_test_time

            overall_score, balanced_score = self.compute_score(test_labels, predicted_labels)
            self.overall_output_scores[train_loader.get_index() - 1] = overall_score
            self.balanced_output_scores[train_loader.get_index() - 1] = balanced_score
            self.confusion_matrices.append(
                pd.DataFrame(metrics.confusion_matrix(test_labels, predicted_labels)).transpose()
            )

            train_data = train_loader.next()
            test_data = test_loader.next()
            index += 1
            print('Step', index, 'finished in', time() - init, 'seconds.',
                  'Overall:', overall_score, 'Balanced score:', balanced_score)

    def compute_score(self, original_labels: np.ndarray, predicted_labels: np.ndarray) -> Tuple[np.float, np.float]:
        return accuracy_score(original_labels, predicted_labels), \
               balanced_accuracy_score(original_labels, predicted_labels)

    def show_results(self):
        print('Mean overall accuracy:', np.mean(self.overall_output_scores))
        print('Mean balanced accuracy:', np.mean(self.balanced_output_scores))

    def get_scores(self):
        return np.mean(self.overall_output_scores), np.mean(self.balanced_output_scores), self.overall_output_scores, self.balanced_output_scores

    def get_times(self):
        return self.total_train_time, self.total_test_time

    def get_confusion_matrices(self):
        return self.confusion_matrices

    # Subclass main methods

    @abc.abstractmethod
    def train(self, values: np.ndarray, labels: np.ndarray):
        raise NotImplementedError('Method not implemented in abstract class')

    @abc.abstractmethod
    def test(self, test_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in abstract class')
