from src.data.dataset import DataLoader
from src.reduce_methods.reduce_method import ReduceMethod


class EmptyReduceMethod(ReduceMethod):

    # Main methods

    def __init__(self, config, output_path, verbose):
        super().__init__(config, output_path, verbose)
        pass

    def run(self, data_loader: DataLoader) -> DataLoader:
        pass
