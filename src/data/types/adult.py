import string
from src.data.dataset import Dataset
import numpy as np
import pandas as pd


class AdultDataset(Dataset):

    def __init__(self, config, verbose):
        self.name = 'adult'
        super(AdultDataset, self).__init__(config, verbose)

    # Main methods

    def get_dataset_name(self) -> string:
        return self.name

    def preprocess_data(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        # TODO implement this
        pass
