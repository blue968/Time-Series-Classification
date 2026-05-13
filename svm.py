import numpy as np
import pandas as pd
from dataloader import TimeSeriesDataLoader
from distances import TimeSeriesKernelFactory
from logging_utils import ExperimentLogger
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def run_svm_experiment(dataset_range=(0, 1), noise_configs=None, metrics=None, dataset_names=None, svm_C=1.0):
    """
    Running SVM Robustness Experiment
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
                    dist_train = factory.compute_distance_matrix(X_train)
                    kernel_train = factory.transform_to_kernel(dist_train)

                    dist_test = factory.compute_distance_matrix(X_test, X_train)
                    kernel_test = factory.transform_to_kernel(dist_test)

                    model = SVC(kernel='precomputed', C=svm_C)
                    model.fit(kernel_train, y_train)

                    y_pred = model.predict(kernel_test)
                    acc = accuracy_score(y_test, y_pred)
                    fold_accuracies.append(acc)
                    print(f"Fold {fold_idx+1} Accuracy: {acc:.4f}")

                avg_acc = np.mean(fold_accuracies)
                logger.log_result(ds_name, n_type, intensity, avg_acc, classifier="SVM", metric=metric)