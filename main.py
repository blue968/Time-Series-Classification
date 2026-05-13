import argparse

from distances import DISTANCES
from ucr_dataset import all_ucr_datasets
from knn import run_1nn_experiment
from svm import run_svm_experiment

CONFIG = {
    # Which experiment to run: '1nn', 'svm', or 'both'
    'experiment': 'both',
    'dataset_range': (7, 8),
    'noise_configs': [
        (t, i) for t in ['jittering', 'missing', 'outlier', 'scaling', 'trend']
        for i in [0.0, 0.05, 0.1, 0.2, 0.4]
    ],
    'metric': DISTANCES,
    'datasets': 'BeetleFly',
    'svm_C': 1.0,
}

def main():
    parser = argparse.ArgumentParser(description='Run time-series experiments')
    parser.add_argument('--experiment', choices=['1nn', 'svm', 'both'], default=CONFIG['experiment'])
    parser.add_argument('--dataset-start', type=int, default=CONFIG['dataset_range'][0])
    parser.add_argument('--dataset-end', type=int, default=CONFIG['dataset_range'][1])
    parser.add_argument('--datasets', type=str, default=CONFIG['datasets'])
    parser.add_argument('--metrics', type=str, default=CONFIG['metric'])
    parser.add_argument('--noise-configs', type=str, default=CONFIG['noise_configs'])
    parser.add_argument('--svm-C', type=float, default=CONFIG['svm_C'])
    args = parser.parse_args()

    if args.experiment in ('1nn', 'both'):
        print('\n=== Running 1-NN Experiment ===')
        run_1nn_experiment(dataset_range=(args.dataset_start, args.dataset_end), dataset_names=args.datasets, noise_configs=args.noise_configs, metrics=args.metrics)

    if args.experiment in ('svm', 'both'):
        print('\n=== Running SVM Experiment ===')
        run_svm_experiment(dataset_range=(args.dataset_start, args.dataset_end), noise_configs=args.noise_configs, metrics=args.metrics, dataset_names=args.datasets)


if __name__ == '__main__':
    main()
