import abc
import numpy as np


class SupervisedAlgorithm:

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'train') and
                callable(subclass.train) and
                hasattr(subclass, 'test') and
                callable(subclass.test) or
                NotImplemented)

    # Main methods

    def __init__(self, config, output_path, verbose):
        pass

    def classify(self, train_values: np.ndarray, train_labels: np.ndarray, test_values: np.ndarray) -> (np.ndarray, np.ndarray):
        # TODO do k fold
        output_train_labels = self.train(train_values, train_labels)
        output_test_labels = self.test(test_values)
        return output_train_labels, output_test_labels

    @abc.abstractmethod
    def train(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def test(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in interface class')
