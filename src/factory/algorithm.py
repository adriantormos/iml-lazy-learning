from src.algorithms.supervised_algorithm import SupervisedAlgorithm
from src.algorithms.types.knn import KNNAlgorithm
from src.algorithms.types.mcnn import ModifiedCondensedKNNAlgorithm
from src.algorithms.types.menn import ModifiedEditedKNNAlgorithm
from src.algorithms.types.drop3 import DROP3KNNAlgorithm


class AlgorithmFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_supervised_algorithm(config, output_path, verbose) -> SupervisedAlgorithm:
        name = config['name']
        if name == 'knn':
            algorithm = KNNAlgorithm(config, output_path, verbose)
        elif name == 'mcnn':
            algorithm = ModifiedCondensedKNNAlgorithm(config, output_path, verbose)
        elif name == 'menn':
            algorithm = ModifiedEditedKNNAlgorithm(config, output_path, verbose)
        elif name == 'drop3':
            algorithm = DROP3KNNAlgorithm(config, output_path, verbose)
        else:
            raise Exception('The supervised algorithm with name ' + name + ' does not exist')
        if issubclass(type(algorithm), SupervisedAlgorithm):
            return algorithm
        else:
            raise Exception('The supervised algorithm does not follow the interface definition')
