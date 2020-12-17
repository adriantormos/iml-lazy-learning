from src.data.dataset import Dataset
from src.data.types.hypothyroid import HypothyroidDataset
from src.data.types.splice import SpliceDataset


class DatasetFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_dataset(config, verbose) -> Dataset:
        name = config['name']
        if name == 'hypothyroid':
            dataset = HypothyroidDataset(config, verbose)
        elif name == 'splice':
            dataset = SpliceDataset(config, verbose)
        else:
            raise Exception('The dataset with name ' + name + ' does not exist')
        if issubclass(type(dataset), Dataset):
            return dataset
        else:
            raise Exception('The dataset does not follow the interface definition')
