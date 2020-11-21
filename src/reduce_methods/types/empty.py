import numpy as np
from src.reduce_methods.reduce_method import ReduceMethod


class EmptyReduceMethod(ReduceMethod):

    # Main methods

    def __init__(self, config, output_path, verbose):
        pass

    def run(self, values: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray):
        pass
