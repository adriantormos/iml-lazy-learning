import argparse
import sys
from src.auxiliary.file_methods import load_json, save_json
from src.factory.dataset import DatasetFactory
from src.factory.algorithm import AlgorithmFactory
from src.factory.reduce_method import ReduceMethodFactory
import random
import numpy as np
from time import time


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
    print('Preparing data')
    initial_time = time()
    dataset = DatasetFactory.select_dataset(data_config, verbose)
    train_loader = dataset.get_train_loader()
    test_loader = dataset.get_test_loader()
    print('Finished preparing data; elapsed time:', time() - initial_time)

    # Run reduction method if required
    if 'reduce_method' in config:
        print('\nReducing data')
        initial_time = time()
        reduce_method_config = config['reduce_method']
        reduce_method = ReduceMethodFactory.select_reduce_method(reduce_method_config, output_path, verbose)
        reduce_method.reduce_data(train_loader)
        print('Finished reducing data; elapsed time:', time() - initial_time)

    # Run algorithm if required
    if 'algorithm' in config:
        print('\nRunning algorithm')
        initial_time = time()
        algorithm_config = config['algorithm']
        algorithm = AlgorithmFactory.select_supervised_algorithm(algorithm_config, output_path, verbose)
        algorithm.classify(train_loader, test_loader)
        algorithm.show_results()
        overall_score, balanced_score, all_scores, all_balanced_scores = algorithm.get_scores()
        confusion_matrices = algorithm.get_confusion_matrices()
        algorithm_total_time = time() - initial_time
        print('Finished running algorithm; elapsed time:', algorithm_total_time)

    # Save config json
    if output_path is not None:
        save_json(output_path + '/config', config)
        if 'algorithm' in config:
            save_json(output_path + '/stats', {'overall_score': overall_score,
                                               'balanced_score': balanced_score,
                                               'all_scores': list(all_scores),
                                               'all_balanced_scores': list(all_balanced_scores),
                                               'execution_time': algorithm_total_time})
            for i, matrix in enumerate(confusion_matrices):
                matrix.to_csv(output_path + '/confusion_matrix_' + str(i) + '.csv', index=False)



if __name__ == '__main__':
    args = parse_arguments()
    # redirect program output if required
    if not args.visualize and args.output_path:
        f = open(args.output_path + '/log.txt', 'w')
        sys.stdout = f
    main(args.config_path, args.output_path, args.visualize, args.verbose)
    if not args.visualize and args.output_path:
        f.close()
