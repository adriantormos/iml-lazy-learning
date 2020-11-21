from src.data.dataset import Dataset
from src.auxiliary.file_methods import load_arff
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class AdultDataset(Dataset):

    # Main methods

    def __init__(self, config, verbose):
        super(AdultDataset, self).__init__(config, verbose)

    def get_preprocessed_data(self) -> (np.ndarray, np.ndarray):
        pass

    def get_preprocessed_dataframe(self) -> pd.DataFrame:
        pass

    # Auxiliary methods

    def preprocess_dataset(self):
        return None
