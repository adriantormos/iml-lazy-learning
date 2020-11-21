from src.reduce_methods.reduce_method import ReduceMethod
from src.reduce_methods.types.empty import EmptyReduceMethod


class ReduceMethodFactory:

    def __init__(self):
        raise Exception('This class can not be instantiated')

    @staticmethod
    def select_reduce_method(config, output_path, verbose) -> ReduceMethod:
        name = config['name']
        if name == 'empty':
            algorithm = EmptyReduceMethod(config, output_path, verbose)
        else:
            raise Exception('The reduce method with name ' + name + ' does not exist')
        if issubclass(type(algorithm), ReduceMethod):
            return algorithm
        else:
            raise Exception('The reduce method does not follow the interface definition')
