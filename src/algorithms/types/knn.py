from collections import Counter

import numpy as np
from scipy.spatial.distance import cdist

from src.algorithms.supervised_algorithm import SupervisedAlgorithm


class KNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.k = config['k']
        try:
            # self.distance_metric = eval(config['distance_metric'] + '_distance_metric')
            self.distance_metric = config['distance_metric']
            if self.distance_metric == 'manhattan':
                self.distance_metric = 'cityblock'
        except KeyError:
            raise Exception('The chosen distance metric does not exist')
        try:
            self.voting = eval(config['voting'] + '_voting_method')
        except:
            raise Exception('The chosen voting method does not exist')
        try:
            self.weighting = eval(config['weighting'] + '_weighting_method')
        except:
            raise Exception('The chosen weighting method does not exist')

    def train(self, values: np.ndarray, labels: np.ndarray):
        print('    Train shape:', values.shape)
        if values.shape[0] < self.k:
            raise Exception('The number of samples of the training set is inferior to the k parameter')
        self.train_values = values
        self.train_labels = labels

    def test(self, test_values: np.ndarray) -> np.ndarray:
        predicted_labels = np.zeros(test_values.shape[0])
        print('    Test shape:', test_values.shape)
        for index, test_value in enumerate(test_values):
            k_close_distances, k_close_labels = self.find_k_close_values(test_value)
            predicted_labels[index] = self.voting(k_close_distances, k_close_labels)
        return predicted_labels

    # Auxiliary methods

    def find_k_close_values(self, test_value: np.ndarray) -> (list, list):
        # pre: self.train_values.shape[0] > k

        distances_to_test_value = cdist(np.array([test_value]), self.train_values, self.distance_metric)[0]
        sorted_args = distances_to_test_value.argsort()[:self.k]
        return distances_to_test_value[sorted_args], self.train_labels[sorted_args]


def majority_voting_method(k_close_distances: list, k_close_labels: list):
    occurrence_count = Counter(k_close_labels)
    return occurrence_count.most_common(1)[0][0]


def equal_weighting_method():
    pass
