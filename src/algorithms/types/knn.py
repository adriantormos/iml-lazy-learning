import math
import random
import numpy as np
from scipy.spatial.distance import cdist

from src.algorithms.supervised_algorithm import SupervisedAlgorithm
from src.algorithms.types.relieff import ReliefF


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
        if config['weighting']['name'] == 'weighted_relieff':
            self.feature_selection = ReliefF(config['weighting'])
        else:
            self.feature_selection = None

    def train(self, values: np.ndarray, labels: np.ndarray):
        print('    Train shape:', values.shape)
        if values.shape[0] < self.k:
            raise Exception('The number of samples of the training set is inferior to the k parameter')
        self.train_values = values
        self.train_labels = labels

        # auxiliary attributes for making easier the voting methods
        all_labels = np.unique(labels)
        self.all_labels = {label:index for index, label in enumerate(all_labels)}
        self.all_labels_inv = {index: label for index, label in enumerate(all_labels)}

        # feature weighting
        if self.feature_selection is not None:
            self.feature_weights = self.feature_selection.run(values, labels)

    def test(self, test_values: np.ndarray) -> np.ndarray:
        predicted_labels = np.zeros(test_values.shape[0])
        print('    Test shape:', test_values.shape)
        for index, test_value in enumerate(test_values):
            k_close_distances, k_close_labels = self.find_k_close_values(test_value)
            predicted_labels[index] = self.voting(k_close_distances, k_close_labels, self.all_labels, self.all_labels_inv)
        return predicted_labels

    # Auxiliary methods

    def find_k_close_values(self, test_value: np.ndarray) -> (list, list):
        # pre: self.train_values.shape[0] > k
        if self.feature_selection is not None:
            distances_to_test_value = cdist(np.expand_dims(test_value*self.feature_weights, axis=0), self.train_values*self.feature_weights, self.distance_metric)[0]
        else:
            distances_to_test_value = cdist(np.expand_dims(test_value, axis=0), self.train_values, self.distance_metric)[0]
        sorted_args = distances_to_test_value.argsort()[:self.k]
        return distances_to_test_value[sorted_args], self.train_labels[sorted_args]


def majority_voting_method(k_close_distances: list, k_close_labels: list, all_labels: dict, all_labels_inv: dict):
    # find the labels with the most apparitions and save their average sum of distances for the possible tie
    labels_count = np.zeros(len(all_labels))
    labels_distance_average = np.zeros(len(all_labels))
    max_count_index = None # store the index of the labels inside the labels_count array with the most apparitions
    max_count_tie = [] # store the indexes of the labels inside the labels_count array with the most apparitions, if there is only one it stores the same as the previous variable
    min_distance_index = None # store the minimum average sum of distances of the labels with the same amount of apparitions (the ones inside the previous variable list)
    for index, k_close_label in enumerate(k_close_labels):
        list_index = all_labels[k_close_label]
        labels_count[list_index] += 1
        labels_distance_average[list_index] += (k_close_distances[index] - labels_distance_average[list_index]) / labels_count[list_index]  # incrementally save the average sum of distances
        if max_count_index is None or labels_count[max_count_index] < labels_count[list_index] or max_count_index == list_index:
            max_count_index = list_index
            max_count_tie = [list_index]
            min_distance_index = list_index
        elif labels_count[max_count_index] == labels_count[list_index]:
            max_count_tie.append(list_index)
            min_distance_index = min_distance_index if labels_distance_average[min_distance_index] <= labels_distance_average[list_index] else list_index

    # if not a tie, return the most common label
    if len(max_count_tie) == 1: return all_labels_inv[max_count_index]

    # if a tie, return the label between the tie labels with the less average distance, if there is also a tie in the distance, we return one of the tie labels
    return all_labels_inv[min_distance_index]


def inverse_distance_weighted_voting_method(k_close_distances: list, k_close_labels: list, all_labels: dict, all_labels_inv: dict):
    # find the labels with the less weighted score
    labels_weighted_score = np.zeros(len(all_labels))
    max_weighted_score_index = None  # store the index of the labels inside the labels_weighted_score array with the highest weighted score
    max_weighted_score_tie = []  # store the indexes of the labels inside the labels_weighted_score array with the highest weighted scores, if there is only one it stores the same as the previous variable
    for index, k_close_label in enumerate(k_close_labels):
        list_index = all_labels[k_close_label]
        labels_weighted_score[list_index] += (1 / k_close_distances[index]) if k_close_distances[index] != 0 else 1000
        if max_weighted_score_index is None or labels_weighted_score[max_weighted_score_index] < labels_weighted_score[list_index] or max_weighted_score_index == list_index:
            max_weighted_score_index = list_index
            max_weighted_score_tie = [list_index]
        elif labels_weighted_score[max_weighted_score_index] == labels_weighted_score[list_index]:
            max_weighted_score_tie.append(list_index)

    # if not a tie, return the label with the minimum sum of weighted votes
    if len(max_weighted_score_tie) == 1: return all_labels_inv[max_weighted_score_index]

    # if a tie, return a random label between the tie labels
    return all_labels_inv[random.choice(max_weighted_score_tie)]


def sheppard_voting_method(k_close_distances: list, k_close_labels: list, all_labels: dict, all_labels_inv: dict):
    # find the labels with the less weighted score
    labels_weighted_score = np.zeros(len(all_labels))
    max_weighted_score_index = None  # store the index of the labels inside the labels_weighted_score array with the highest weighted score
    max_weighted_score_tie = []  # store the indexes of the labels inside the labels_weighted_score array with the highest weighted scores, if there is only one it stores the same as the previous variable
    for index, k_close_label in enumerate(k_close_labels):
        list_index = all_labels[k_close_label]
        labels_weighted_score[list_index] += math.exp(-k_close_distances[index])
        if max_weighted_score_index is None or labels_weighted_score[max_weighted_score_index] < labels_weighted_score[list_index] or max_weighted_score_index == list_index:
            max_weighted_score_index = list_index
            max_weighted_score_tie = [list_index]
        elif labels_weighted_score[max_weighted_score_index] == labels_weighted_score[list_index]:
            max_weighted_score_tie.append(list_index)

    # if not a tie, return the label with the minimum sum of weighted votes
    if len(max_weighted_score_tie) == 1: return all_labels_inv[max_weighted_score_index]

    # if a tie, return a random label between the tie labels
    return all_labels_inv[random.choice(max_weighted_score_tie)]


def equal_weighting_method():
    pass
