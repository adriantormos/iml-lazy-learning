import abc
from src.data.dataset import DataLoader


class ReduceMethod:

    @classmethod
    def __subclasshook__(cls, subclass):  # to check that the subclasses follow the interface
        return (hasattr(subclass, 'run') and
                callable(subclass.run) or
                NotImplemented)

    # Main methods

    def __init__(self, config, output_path, verbose):
        pass

    def reduce_data(self, train_loader: DataLoader) -> DataLoader:
        return self.run(train_loader)

    @abc.abstractmethod
    def run(self, data_loader: DataLoader) -> DataLoader:
        raise NotImplementedError('Method not implemented in abstract class')
