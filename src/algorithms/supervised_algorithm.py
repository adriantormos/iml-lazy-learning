import abc
import numpy as np
from src.data.dataset import DataLoader
from sklearn.metrics import accuracy_score
from time import time


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
        pass

    def classify(self, train_loader: DataLoader, test_loader: DataLoader):
        self.output_scores = np.zeros(train_loader.get_length())
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

            self.train(train_values, train_labels)
            predicted_labels = self.test(test_values)

            score = self.compute_score(test_labels, predicted_labels)
            self.output_scores[train_loader.get_index()-1] = score

            train_data = train_loader.next()
            test_data = test_loader.next()
            index += 1
            print('Step', index, 'finished in', time() - init, 'seconds. Score:', score)

    def compute_score(self, original_labels: np.ndarray, predicted_labels: np.ndarray) -> np.float:
        return accuracy_score(original_labels, predicted_labels)

    def show_results(self):
        print('Mean accuracy:', np.mean(self.output_scores))

    # Subclass main methods

    @abc.abstractmethod
    def train(self, values: np.ndarray, labels: np.ndarray):
        raise NotImplementedError('Method not implemented in abstract class')

    @abc.abstractmethod
    def test(self, test_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Method not implemented in abstract class')
