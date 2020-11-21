import abc
import numpy as np


class ReduceMethod:

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    # Main methods

    def __init__(self, config, output_path, verbose):
        pass

    def reduce_data(self, values: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        return self.run(values, labels)

    @abc.abstractmethod
    def run(self, values: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        raise NotImplementedError('Method not implemented in interface class')
