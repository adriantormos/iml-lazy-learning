import numpy as np
from scipy.spatial.distance import cdist


class ReliefF:

    def __init__(self, config):
        self.n_iterations = config['n_iterations']
        self.k = config['nearest_values']  # nearest hits or misses to find
        self.distance_metric = config['distance_metric']

    def run(self, values, labels):
        feature_weights = np.zeros(values.shape[1])

        label_classes, label_counts = np.unique(labels, return_counts=True)
        label_classes_index = {label: index for index, label in enumerate(label_classes)}
        label_classes_probabilities = {label: label_counts[index]/len(labels) for index, label in enumerate(label_classes)}

        for _ in range(self.n_iterations):
            # select random instance
            instance_index = np.random.randint(0, values.shape[0])
            instance_class = labels[instance_index]

            # find the distance between this instance and the rest of them
            distances_to_instance = cdist(np.array([values[instance_index]]), values, self.distance_metric)[0]

            # sort the labels and values regarding the distances
            sorted_args = distances_to_instance.argsort()
            sorted_labels = labels[sorted_args]
            sorted_values = values[sorted_args]

            # find nearest hits for the instance class and nearest misses for the rest of classes
            total_found_count = np.zeros(len(label_classes_index))
            total_found = 0  # counter that tells which number of classes have accomplish their goal, if total_found == number_classes all classes have finished

            index = 1  # do not count the same instance
            denominator = self.n_iterations * self.k
            while index < len(sorted_labels) and total_found < len(label_classes_index):
                label = sorted_labels[index]
                if total_found_count[label_classes_index[label]] < self.k:
                    total_found_count[label_classes_index[label]] += 1
                    score = np.divide(np.abs(np.subtract(values[instance_index], sorted_values[index])),denominator)
                    if instance_class == label:
                        # update weights as hit
                        feature_weights -= score
                    else:
                        # update weights as miss
                        feature_weights += (label_classes_probabilities[label] / (1 - label_classes_probabilities[instance_class])) * score
                    if total_found_count[label_classes_index[label]] == self.k:
                        total_found += 1
                index += 1

        # normalize the feature weights to use them in the knn algorithm
        feature_weights = (feature_weights - np.min(feature_weights))/np.ptp(feature_weights)

        return feature_weights
