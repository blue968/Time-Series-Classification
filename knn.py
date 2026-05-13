import numpy as np
import pandas as pd
from dataloader import TimeSeriesDataLoader
from distances import TimeSeriesKernelFactory
from logging_utils import ExperimentLogger
from sklearn.metrics import accuracy_score

def run_1nn_experiment(dataset_range=(0, 1), noise_configs=None, metrics=None, dataset_names=None):
    """
    Running 1-NN Robustness Experiment
    """
    loader = TimeSeriesDataLoader(n_splits=5)
    logger = ExperimentLogger()

    if metrics is None:
        from distances import DISTANCES
        metrics = DISTANCES
    elif isinstance(metrics, str):
        metrics = [metrics]

    if dataset_names is not None:
        datasets = dataset_names.split(',') if isinstance(dataset_names, str) else list(dataset_names)
    else:
        datasets = loader.get_dataset_by_range(*dataset_range)

    for ds_name in datasets:
        for metric in metrics:
            for n_type, intensity in noise_configs:
                print(f"\n--- Dataset={ds_name} | noise={n_type} ({intensity}) | metric={metric} ---")
                folds_list = list(loader.get_noisy_folds(ds_name, noise_type=n_type, intensity=intensity))
                fold_accuracies = []

                for fold_idx, (X_train, X_test, y_train, y_test) in enumerate(folds_list):
                    factory = TimeSeriesKernelFactory(metric=metric)
                    dist_matrix = factory.compute_distance_matrix(X_test, X_train)

                    nn_indices = np.argmin(dist_matrix, axis=1)
                    y_pred = y_train[nn_indices]

                    acc = accuracy_score(y_test, y_pred)
                    fold_accuracies.append(acc)
                    print(f"Fold {fold_idx+1} Accuracy: {acc:.4f}")

                avg_acc = np.mean(fold_accuracies)
                logger.log_result(ds_name, n_type, intensity, avg_acc, classifier="1-NN", metric=metric)