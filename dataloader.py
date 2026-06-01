import numpy as np
from aeon.datasets import load_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from ucr_dataset import all_ucr_datasets, dataset_domains

class TimeSeriesDataLoader:
    def __init__(self, n_splits=5, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.le = LabelEncoder()
        self.all_ucr_datasets = all_ucr_datasets
        self.dataset_domains = dataset_domains

    def get_dataset_by_range(self, k, t):
        selected = self.all_ucr_datasets[k:t]
        return selected

    def _load_single_dataset(self, name):
        try:
            X, y = load_classification(name, split=None)
            domain = self.dataset_domains.get(name, "unknown")
            print(f"Name: {name}")
            print(f"Domain: {domain}")
            print(f"Samples: {X.shape[0]}")
            print(f"Classes: {len(np.unique(y))}")
            print(f"Length: {X.shape[2]}")
            print("-" * 50)
            return X.astype(np.float64), y
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return None, None

    # --- Noise ---
    
    def _add_jittering(self, X, sigma):
        """
        Add Additive Gaussian White Noise (Jittering)
        - Adds random Gaussian noise with mean=0 and std=sigma to each data point
        - Common in real-world sensor data and signal processing
        - Parameters: sigma controls noise magnitude
        """
        rng = np.random.default_rng(self.random_state)
        return X + rng.normal(0, sigma, X.shape)

    def _add_outliers(self, X, prob=0.05, magnitude=5):
        """
        Add Outlier Noise (Sparse Impulse Noise)
        - Randomly replaces (prob) proportion of data points with large deviations
        - Each outlier is offset by: ±(magnitude × signal_std)
        - Simulates sudden spikes, sensor failures, or anomalous events
        - Parameters: prob (0-1) is outlier frequency, magnitude scales deviation size
        """
        X_noisy = X.copy()
        rng = np.random.default_rng(self.random_state)
        mask = rng.random(X.shape) < prob
        X_noisy[mask] += magnitude * X.std() * rng.choice([-1, 1], size=np.sum(mask))
        return X_noisy

    def _add_missing_values(self, X, prob=0.1):
        """
        Add Missing Values (Zero Masking)
        - Randomly replaces (prob) proportion of data points with zeros
        - Simulates data loss, sensor dropout, or incomplete recording
        - Note: Values are set to 0 (not NaN), assuming zero is a valid baseline
        - Parameters: prob (0-1) is fraction of data points to zero out
        """
        X_noisy = X.copy()
        mask = np.random.choice([0, 1], size=X.shape, p=[prob, 1-prob])
        return X_noisy * mask

    # --- Multi-fold ---

    def get_noisy_folds(self, dataset_name, noise_type='jittering', intensity=0.1):
        X, y = self._load_single_dataset(dataset_name)
        if X is None: return
        
        y_enc = self.le.fit_transform(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        noise_funcs = {
            'jittering': self._add_jittering,
            'outlier': self._add_outliers,
            'missing': self._add_missing_values
        }

        if noise_type not in noise_funcs:
            available = ", ".join(sorted(noise_funcs))
            raise ValueError(f"Unknown noise_type '{noise_type}'. Available noise types: {available}")

        noise_func = noise_funcs[noise_type]

        for train_idx, test_idx in skf.split(X, y_enc):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_enc[train_idx], y_enc[test_idx]
            
            if intensity == 0:
                X_train_n = X_train
                X_test_n = X_test
            else:
                X_train_n = noise_func(X_train, intensity)
                X_test_n = noise_func(X_test, intensity)
            
            yield X_train_n, X_test_n, y_train, y_test
