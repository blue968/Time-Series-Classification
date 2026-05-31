import numpy as np
import aeon.distances

DISTANCES_BY_CATEGORY = {
    "lock_step": [
        "euclidean", "manhattan", "minkowski"
    ],
    "sliding": [
        "sbd"
    ],
    "elastic": [
        "dtw", "wdtw", "adtw"
    ],
    "feature_based": [
        "shape_dtw"
    ],
    "edit_based": [
        "twe", "msm", "erp", "edr"
    ],
    "alignment": [
        "lcss"
    ],
    "embedding": [
        "ddtw", "wddtw"
    ]
}

DISTANCES = [dist for sublist in DISTANCES_BY_CATEGORY.values() for dist in sublist]

class TimeSeriesKernelFactory:
    """
    Factory that uses aeon.distances implementations when available.
    """
    def __init__(self, metric=None, **kwargs):
        self.metric = metric
        self.metric_params = kwargs or {}

        try:
            self.dist_func = aeon.distances.get_distance_function(metric)
            print(f"Successfully loaded aeon metric: {metric}")
        except Exception as e:
            print(f"Warning: Could not load metric '{metric}' from aeon. Falling back to Euclidean.")
            self.dist_func = aeon.distances.get_distance_function("euclidean")

    def compute_distance_matrix(self, X1, X2=None):
        """
        Compute pairwise distance matrix with aeon's vectorized implementation.

        Falls back to the pair-by-pair implementation for metrics or input shapes
        that are not accepted by aeon.distances.pairwise_distance.
        """
        X2 = X1 if X2 is None else X2
        try:
            return aeon.distances.pairwise_distance(
                X1,
                X2,
                metric=self.metric,
                **self.metric_params,
            )
        except Exception as e:
            print(f"Warning: pairwise_distance failed for '{self.metric}' ({e}). Falling back to pairwise loop.")
            return self._compute_distance_matrix_loop(X1, X2)

    def _compute_distance_matrix_loop(self, X1, X2):
        """
        Slow fallback for metrics or input shapes not supported by pairwise_distance.
        """
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        dist_mat = np.zeros((n1, n2))
        params = self.metric_params

        for i in range(n1):
            a = self._prepare_input(X1[i])
            for j in range(n2):
                b = self._prepare_input(X2[j])
                try:
                    dist_mat[i, j] = self.dist_func(a, b, **params)
                except Exception:
                    dist_mat[i, j] = self.dist_func(a, b)
        return dist_mat

    def _prepare_input(self, series):
        """
        Transform input to the shape expected by aeon.
        """
        s = np.asarray(series)
        if s.ndim == 2 and s.shape[0] == 1:
            return s[0]
        return s
        
    def transform_to_kernel(self, dist_matrix, gamma=None):
        """
        Map distances to an RBF-like kernel matrix. Ensures positive-definiteness
        by shifting eigenvalues slightly when necessary.
        """
        d_sq = dist_matrix ** 2
        if gamma is None:
            valid_d_sq = d_sq[d_sq > 0]
            median_sq = np.median(valid_d_sq) if len(valid_d_sq) > 0 else 1.0
            gamma = 1.0 / (2 * median_sq) if median_sq != 0 else 1.0
        kernel_matrix = np.exp(-gamma * d_sq)
        
        rows, cols = kernel_matrix.shape
        if rows == cols:
            try:
                min_eig = np.min(np.real(np.linalg.eigvals(kernel_matrix)))
                if min_eig < 0:
                    kernel_matrix += (abs(min_eig) * 1.1 + 1e-8) * np.eye(rows)
            except Exception:
                kernel_matrix += 1e-6 * np.eye(rows)

        return kernel_matrix
