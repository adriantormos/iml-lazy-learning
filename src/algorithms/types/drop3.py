
import numpy as np
from scipy.spatial.distance import cdist
from src.algorithms.supervised_algorithm import SupervisedAlgorithm
from src.algorithms.types.knn import KNNAlgorithm

class DROP3KNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.knn = KNNAlgorithm({**config, 'k': config['k']+1}, output_path, False)
        self.knn2 = KNNAlgorithm(
            {'k': config['k']+1, 'distance_metric': config['distance_metric'],
             'voting': 'majority', "weighting": {"name": "equal"}
             },
            output_path,
            False
        )
        self.mode = config['mode'] if 'mode' in config else 'drop3'
        self.k = config['k']
        try:
            # self.distance_metric = eval(config['distance_metric'] + '_distance_metric')
            self.distance_metric = config['distance_metric']
            if self.distance_metric == 'manhattan':
                self.distance_metric = 'cityblock'
        except KeyError:
            raise Exception('The chosen distance metric does not exist')

    def train(self, values: np.ndarray, labels: np.ndarray):
        if self.verbose:
            print('    Train shape:', values.shape)
        if values.shape[0] < self.k:
            raise Exception('The number of samples of the training set is inferior to the k parameter')

        if self.verbose:
            print('   ', 'Mode:', self.mode)

        # (value, position in original training set)
        definitive_samples = list(zip(values.copy(), list(range(labels.shape[0]))))
        definitive_labels = list(zip(labels.copy(), list(range(labels.shape[0]))))

        associated_samples = {}  # A (associated samples lists)
        nearest_neighbors = {}   # V (nearest neighbors lists,
                                 #    each list is tuple (neighbors' IDs, distance to 1st neighbour, predicted label))

        if self.verbose:
            print('   ', 'Obtaining initial nearest neighbors for', values.shape[0], 'samples')

        self.knn.train(values, labels)
        # For x_i in S
        for i, instance in enumerate(values):
            # Find nearest neighbors of x_i, ordered by distance in ascending order
            neighbors_distances, neighbors_labels, neighbors_pos = self.knn.find_k_close_values_with_position(instance)
            neighbors_labels = neighbors_labels[1:]
            neighbors_distances = neighbors_distances[1:]
            neighbors_pos = neighbors_pos[1:]
            # Find predicted label_i via default KNN with the point's neighbors
            unique_labels, counts_labels = np.unique(neighbors_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts_labels)]
            # Fill V(x_i) (nearest neighbors vector)
            nearest_neighbors[i] = (neighbors_pos, neighbors_distances[0], predicted_label)

            # For a_j in V(x_i), add x_i to A(a_j)
            # (Add x_i to the associated samples list of each of its neighbors)
            for pos in neighbors_pos:
                if pos not in associated_samples:
                    associated_samples[pos] = []
                associated_samples[pos].append(i)

        # DROP3-ONLY SECTION
        if self.mode == 'drop3':
            if self.verbose:
                print('   ', 'Dropping misclassified samples')
            # For x_i in S
            deleted_instances = 0
            for i, instance in enumerate(values):
                # If predicted label_i is wrong, delete x_i from S
                # Also delete A(x_i) as it won't be necessary then
                if nearest_neighbors[i][2] != labels[i]:
                    # Adjust position to delete to the amount of already deleted instances
                    # As i only grows in this loop, this will work
                    definitive_samples = np.delete(definitive_samples, i-deleted_instances, axis=0)
                    definitive_labels = np.delete(definitive_labels, i-deleted_instances, axis=0)
                    deleted_instances += 1
                    if i in associated_samples:
                        del associated_samples[i]

        definitive_labels = np.array(definitive_labels)
        definitive_samples = np.array(definitive_samples)

        if self.verbose:
            print('   ', 'Sorting', definitive_samples.shape[0], 'samples by distance to nearest neighbor')
        # Sort S by distance to nearest neighbor
        zipped_list = list(zip(definitive_samples, definitive_labels))
        zipped_list.sort(key=lambda x: nearest_neighbors[x[0][1]][1], reverse=True)
        # Unzip the auxiliary zipped list
        definitive_samples, definitive_labels = list(zip(*zipped_list))
        definitive_samples = list(definitive_samples)
        definitive_labels = list(definitive_labels)
        if self.verbose:
            print('   ', 'Deleting samples that decrease performance')
        # For x_i in S
        # (sample has the features, original_position the ID/position in original vector)
        deleted_instances = 0
        for i, element in enumerate(definitive_samples.copy()):
            sample, original_position = element
            correctly_predicted_with_sample = 0
            correctly_predicted_without_sample = 0
            # For a_j in A(x_i)
            # (associated_sample is an ID/position in original vector for a_j)
            if original_position in associated_samples:
                for associated_sample in associated_samples[original_position]:
                    # Check if a_j classified correctly with x_i
                    if labels[associated_sample] == nearest_neighbors[associated_sample][2]:
                        correctly_predicted_with_sample += 1

                    # Check if a_j classified correctly without x_i
                    # Get V(a_j) without x_i
                    associated_samples_neighbors = nearest_neighbors[associated_sample][0]
                    associated_samples_neighbors_without_sample = associated_samples_neighbors[
                        associated_samples_neighbors != original_position
                    ]
                    if not associated_samples_neighbors_without_sample.size == 0:
                        # For v_k in V(a_j) get its label
                        associated_samples_neighbors_labels = np.array([nearest_neighbors[pos][2]
                                                                        for pos in associated_samples_neighbors_without_sample])
                        # Find predicted label for a_j via default KNN with the point's neighbors
                        unique_labels, counts_labels = np.unique(associated_samples_neighbors_labels, return_counts=True)
                        predicted_label = unique_labels[np.argmax(counts_labels)]
                        if labels[associated_sample] == predicted_label:
                            correctly_predicted_without_sample += 1

            # If without x_i, more or the same a_j are classified correctly, remove it from S
            if correctly_predicted_without_sample >= correctly_predicted_with_sample:
                # We adjust for the deleted elements just as in the other previous loop
                del definitive_samples[i-deleted_instances]
                del definitive_labels[i-deleted_instances]
                deleted_instances += 1

                if original_position in associated_samples:
                    # For a_j in A(x_i)
                    for associated_sample in associated_samples[original_position]:
                        unzipped_definitive_samples, _ = list(zip(*definitive_samples))
                        unzipped_definitive_labels, _ = list(zip(*definitive_labels))
                        self.knn.train(np.array(unzipped_definitive_samples), np.array(unzipped_definitive_labels))
                        # Find again the k+1 nearest neighbors of a_j, now without x_i
                        neighbors_distances, neighbors_labels, neighbors_pos = self.knn.find_k_close_values_with_position(
                            values[associated_sample]
                        )
                        neighbors_labels = neighbors_labels[1:]
                        neighbors_distances = neighbors_distances[1:]
                        neighbors_pos = neighbors_pos[1:]
                        # Find predicted label of a_j via default KNN with the point's neighbors
                        unique_labels, counts_labels = np.unique(neighbors_labels, return_counts=True)
                        predicted_label = unique_labels[np.argmax(counts_labels)]
                        # Fill V(a_j)
                        nearest_neighbors[associated_sample] = (neighbors_pos, neighbors_distances[0], predicted_label)

                    del associated_samples[original_position]

        # Generate the final KNN trained instance with the final set S
        unzipped_definitive_samples, _ = list(zip(*definitive_samples))
        unzipped_definitive_labels, _ = list(zip(*definitive_labels))
        unzipped_definitive_samples = np.array(unzipped_definitive_samples)
        unzipped_definitive_labels = np.array(unzipped_definitive_labels)
        if self.verbose:
            print('   ', 'Final training set size:', unzipped_definitive_samples.shape[0])
        self.knn.train(unzipped_definitive_samples, unzipped_definitive_labels)

    def test(self, test_values: np.ndarray) -> np.ndarray:
        return self.knn.test(test_values)
