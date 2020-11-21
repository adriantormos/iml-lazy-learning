import argparse
import sys
from src.auxiliary.file_methods import load_json, save_json
from src.factory.dataset import DatasetFactory
from src.factory.algorithm import AlgorithmFactory
from src.factory.reduce_method import ReduceMethodFactory
from src.visualize import show_charts
import random
import numpy as np


def parse_arguments():
    def parse_bool(s: str):
        if s.casefold() in ['1', 'true', 'yes']:
            return True
        if s.casefold() in ['0', 'false', 'no']:
            return False
        raise ValueError()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', help="Path to config file", required=True)
    parser.add_argument('--output_path', default=None, help="Path to output directory", required=False)
    parser.add_argument('--visualize', default=True, type=parse_bool, help="Move standard output to a log file",
                        required=False)
    parser.add_argument('--verbose', default=True, type=parse_bool, help="Show more info", required=False)
    args = parser.parse_args()

    # TODO check the first path is a file and a json
    # TODO check the second path is a directory
    # TODO check config file correctness

    return args


def main(config_path: str, output_path: str, visualize: bool, verbose: bool):

    # Load configuration
    config = load_json(config_path)
    data_config = config['data']
    charts_config = config['charts']

    # Set up
    random.seed(config['manual_seed'])
    np.random.seed(config['manual_seed'])

    # Load and prepare data
    dataset = DatasetFactory.select_dataset(data_config, verbose)
    values, labels = dataset.get_preprocessed_data()
    values, labels = dataset.prepare(values, labels)

    # Run reduction method if required
    if 'reduce_method' in config:
        reduce_method_config = config['reduce_method']
        reduce_method = ReduceMethodFactory.select_reduce_method(reduce_method_config, output_path, verbose)
        reduce_method.reduce_data(values, labels)

    # Run algorithm if required
    if 'algorithm' in config:
        algorithm_config = config['algorithm']
        algorithm = AlgorithmFactory.select_supervised_algorithm(algorithm_config, output_path, verbose)
        output_train_labels, output_test_labels = algorithm.classify(values, labels, None)

        # TODO finish this
        #show_charts(charts_config, output_path, values, labels, output_labels, visualize, dataset.get_preprocessed_dataframe(), verbose)
        #if output_path is not None:
        #    np.save(output_path + '/predicted_labels', output_labels)
    else:
        show_charts(charts_config, output_path, values, labels, None, visualize, dataset.get_preprocessed_dataframe(), verbose)


    # Save config json
    if output_path is not None:
        save_json(output_path + '/config', config)


if __name__ == '__main__':
    args = parse_arguments()
    # redirect program output
    if not args.visualize and args.output_path:
        f = open(args.output_path + '/log.txt', 'w')
        sys.stdout = f
    main(args.config_path, args.output_path, args.visualize, args.verbose)
    if not args.visualize and args.output_path:
        f.close()
