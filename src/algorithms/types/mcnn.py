
import numpy as np
from scipy.spatial.distance import cdist
from src.algorithms.supervised_algorithm import SupervisedAlgorithm
from src.algorithms.types.knn import KNNAlgorithm

class ModifiedCondensedKNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        self.knn = KNNAlgorithm(config, output_path, False)
        self.knn.k = 1
        self.k = config['k']
        try:
            # self.distance_metric = eval(config['distance_metric'] + '_distance_metric')
            self.distance_metric = config['distance_metric']
            if self.distance_metric == 'manhattan':
                self.distance_metric = 'cityblock'
        except KeyError:
            raise Exception('The chosen distance metric does not exist')
        # try:
        #     self.voting = eval(config['voting'] + '_voting_method')
        # except:
        #     raise Exception('The chosen voting method does not exist')
        # try:
        #     self.weighting = eval(config['weighting'] + '_weighting_method')
        # except:
        #     raise Exception('The chosen weighting method does not exist')

    def train(self, values: np.ndarray, labels: np.ndarray):

        def _generate_prototypes(misclassified_samples, misclassified_labels, knn):
            prototype_list = np.empty(shape=(0, values.shape[1]))
            prototype_labels = np.array([])
            well_classified_samples = misclassified_samples.copy()
            well_classified_labels = misclassified_labels.copy()
            # Generate prototypes (construct Q)
            while misclassified_samples.size != 0:
                # Compute prototypes for this iteration (P_tj)
                this_iteration_prototypes = []
                this_iteration_prototypes_labels = []
                for j in range(max_label+1):
                    # Filter samples of class j
                    misclassified_j_samples = well_classified_samples[well_classified_labels == j]
                    if misclassified_j_samples.size != 0:
                        # Compute the centroid of class j, C_j
                        centroid_j = np.mean(misclassified_j_samples, axis=0)
                        # Get the closest point to C_j, our prototype P_j for class j
                        distances_to_centroid_j = cdist(np.array([centroid_j]),
                                                        misclassified_j_samples,
                                                        self.distance_metric)[0]
                        position_of_prototype_j = distances_to_centroid_j.argmin()
                        prototype_j = misclassified_j_samples[position_of_prototype_j]

                        this_iteration_prototypes.append(prototype_j)
                        this_iteration_prototypes_labels.append(j)
                        # Remove P_j from S_t
                        # misclassified_samples = np.delete(misclassified_samples, position_of_prototype_j, axis=0)
                        # misclassified_labels = np.delete(misclassified_labels, position_of_prototype_j)

                this_iteration_prototypes = np.array(this_iteration_prototypes)
                this_iteration_prototypes_labels = np.array(this_iteration_prototypes_labels)

                # Test prototypes
                knn.train(this_iteration_prototypes, this_iteration_prototypes_labels)
                knn_returned_labels = knn.test(well_classified_samples)

                # Filter well classified samples (compute new S_t)
                classification_index = well_classified_labels == knn_returned_labels
                misclassified_samples = well_classified_samples[~classification_index]
                well_classified_samples = well_classified_samples[classification_index]
                well_classified_labels = well_classified_labels[classification_index]

                # Update Q
                already_a_prototype_index = np.apply_along_axis(
                    lambda x: (np.array([x]) == prototype_list).all(axis=1).any(),
                    1, this_iteration_prototypes)
                this_iteration_prototypes_labels = this_iteration_prototypes_labels[~already_a_prototype_index]
                this_iteration_prototypes = this_iteration_prototypes[~already_a_prototype_index]
                prototype_list = np.append(prototype_list, this_iteration_prototypes, axis=0)
                prototype_labels = np.append(prototype_labels, this_iteration_prototypes_labels)
                unique_prototypes = np.unique(prototype_list, axis=0)

            return prototype_list, prototype_labels

        if self.verbose:
            print('    Train shape:', values.shape)
        if values.shape[0] < self.k:
            raise Exception('The number of samples of the training set is inferior to the k parameter')

        max_label: int = np.max(labels)
        # set S_t
        _misclassified_samples = values.copy()
        _misclassified_labels = labels.copy()
        # set Q
        _prototype_list = np.empty(shape=(0, values.shape[1]))
        _prototype_labels = np.array([])
        while _misclassified_samples.size != 0 and np.shape(_prototype_labels)[0] < np.shape(labels)[0]:
            add_to_prototype_list, add_to_prototype_labels = _generate_prototypes(_misclassified_samples,
                                                                                  _misclassified_labels,
                                                                                  self.knn)
            _prototype_list = np.append(_prototype_list, add_to_prototype_list, axis=0)
            _prototype_labels = np.append(_prototype_labels, add_to_prototype_labels)

            if self.verbose:
                print('   ', 'Test using',
                      np.shape(_prototype_list)[0], 'prototypes on all testing set:', end='')

            self.knn.train(_prototype_list, _prototype_labels)
            _knn_returned_labels = self.knn.test(values)

            _classification_index = _knn_returned_labels != labels
            _is_prototype_index = np.apply_along_axis(lambda x: (np.array([x]) == _prototype_list).all(axis=1).any(),
                                                      1, values)
            _prototype_indices = np.nonzero(_is_prototype_index)
            _misclassified_samples = values[_classification_index]
            _misclassified_labels = labels[_classification_index]

            if self.verbose:
                print('', str(np.shape(_misclassified_samples)[0]) + '/' + str(np.shape(values)[0]),
                      'misclassified samples')

        if self.verbose:
            print('   ', 'Deleteing unused prototypes')
        used_instances = []
        self.knn.train(_prototype_list, _prototype_labels)
        for instance in values:
            # k is always 1
            _, _, used_instance = self.knn.find_k_close_values_with_position(instance)
            used_instances.extend(list(used_instance))
        used_instances = np.array(np.unique(used_instances))
        _prototype_list = _prototype_list[used_instances]
        _prototype_labels = _prototype_labels[used_instances]

        if self.verbose:
            print('   ', 'Final training set size:', _prototype_list.shape[0])
        self.knn.train(_prototype_list, _prototype_labels)

    def test(self, test_values: np.ndarray) -> np.ndarray:
        self.knn.k = self.k
        return self.knn.test(test_values)

