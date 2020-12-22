
import numpy as np
from scipy.spatial.distance import cdist
from src.algorithms.supervised_algorithm import SupervisedAlgorithm
from src.algorithms.types.knn import KNNAlgorithm

class ModifiedEditedKNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.knn = KNNAlgorithm(config, output_path, verbose)
        self.k = config['k']
        try:
            # self.distance_metric = eval(config['distance_metric'] + '_distance_metric')
            self.distance_metric = config['distance_metric']
            if self.distance_metric == 'manhattan':
                self.distance_metric = 'cityblock'
        except KeyError:
            raise Exception('The chosen distance metric does not exist')

    def train(self, values: np.ndarray, labels: np.ndarray, _):
        if self.verbose:
            print('    Train shape:', values.shape)
        if values.shape[0] < self.k:
            raise Exception('The number of samples of the training set is inferior to the k parameter')

        typical_samples = []
        typical_labels = []
        set_length = values.shape[0]
        for i in range(set_length):
            if i % 1000 == 0:
                print('   ', 'Computing distances for sample', i)
            candidate_value = values[i]
            candidate_label = labels[i]
            k_l_closest_values, k_l_closest_labels = self.find_k_l_close_values(candidate_value, values, labels)
            if np.all(k_l_closest_labels == candidate_label):
                typical_samples.append(candidate_value)
                typical_labels.append(candidate_label)

        self.knn.train(np.array(typical_samples), np.array(typical_labels), True)
        if self.verbose:
            print('   ', 'Final training set size:', np.array(typical_samples).shape[0])

    def test(self, test_values: np.ndarray) -> np.ndarray:
        return self.knn.test(test_values)

    def find_k_l_close_values(self, point: np.ndarray, values, labels) -> (list, list):
        # pre: self.train_values.shape[0] > k
        distances_to_test_value = cdist(np.array([point]), values, self.distance_metric)[0]

        k = self.k+1
        sorted_distances = distances_to_test_value.argsort()
        while distances_to_test_value[sorted_distances[k]] == distances_to_test_value[sorted_distances[k+1]]:
            k += 1

        return values[sorted_distances[1:k]], labels[sorted_distances[1:k]]
