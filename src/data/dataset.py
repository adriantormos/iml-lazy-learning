import abc
import pandas as pd
import numpy as np
from src.auxiliary.preprocessing_methods import shuffle_in_unison


class Dataset(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'get_preprocessed_data') and
                callable(subclass.get_preprocessed_data) and
                hasattr(subclass, 'get_preprocessed_dataframe') and
                callable(subclass.get_preprocessed_dataframe) or
                NotImplemented)

    # Main methods

    def __init__(self, config, verbose):
        self.config = config

    @abc.abstractmethod
    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        raise NotImplementedError('Method not implemented in interface class')

    @abc.abstractmethod
    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        raise NotImplementedError('Method not implemented in interface class')

    def prepare(self, values: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        prepare_steps = self.config['prepare']
        for prepare_step in prepare_steps:
            name = prepare_step['name']
            if name == 'shuffle':
                shuffle_in_unison(values, labels)
            else:
                raise Exception('The prepare step with name ' + name + ' does not exist')
        return values, labels

    def split(self, values: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        pass

    # Auxiliary methods
