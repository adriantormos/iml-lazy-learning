from src.algorithms.supervised_algorithm import SupervisedAlgorithm
import numpy as np


class KNNAlgorithm(SupervisedAlgorithm):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)

    def train(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        pass

    def test(self, values: np.ndarray) -> np.ndarray:
        pass

    # Auxiliary methods
