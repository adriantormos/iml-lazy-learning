from src.algorithms.supervised_algorithm import SupervisedAlgorithm
import bisect
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine


class KNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.k = config['k']
        try:
            self.distance_metric = eval(config['distance_metric'] + '_distance_metric')
        except:
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

    def find_k_close_values(self, test_value) -> (list, list):
        # pre: self.train_values.shape[0] > k

        # lists to maintain in ascending order the k closer values
        k_close_distances = []
        k_close_labels = []

        # initialize first k close distances
        for index in range(self.k):
            distance_score = self.distance_metric(self.train_values[index], test_value)
            pos_to_insert = bisect.bisect(k_close_distances, distance_score)
            k_close_distances.insert(pos_to_insert, distance_score)
            k_close_labels.insert(pos_to_insert, self.train_labels[index])

        # find the k close values
        max_k_score = k_close_distances[-1]
        for index, train_value in enumerate(self.train_values[self.k:]):
            distance_score = self.distance_metric(train_value, test_value)
            if distance_score < max_k_score:
                del k_close_distances[-1]
                del k_close_labels[-1]
                pos_to_insert = bisect.bisect(k_close_distances, distance_score)
                k_close_distances.insert(pos_to_insert, distance_score)
                k_close_labels.insert(pos_to_insert, self.train_labels[index + self.k])
                max_k_score = k_close_distances[-1]

        return k_close_distances, k_close_labels


# Static util functions for the knn algorithm

def euclidean_distance_metric(x: np.ndarray, y: np.ndarray):
    return np.linalg.norm(x-y)


def manhattan_distance_metric(x: np.ndarray, y: np.ndarray):
    return np.sum(np.abs(x-y))


def cosine_distance_metric(x: np.ndarray, y: np.ndarray):
    return cosine(x, y)


def majority_voting_method(k_close_distances: list, k_close_labels: list):
    occurrence_count = Counter(k_close_labels)
    return occurrence_count.most_common(1)[0][0]


def equal_weighting_method():
    pass
