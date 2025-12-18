import warnings
from typing import Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import log_loss


class GaussianProcessSwapModel:
    """
    Gaussian Process classifier wrapper used to score client-agent swap proposals.
    Fits on the entire replay buffer for an agent and returns class probabilities.
    """

    def __init__(self, input_dim: int, base_length_scale: float = 1.0):
        self.input_dim = input_dim
        length_scales = np.full(input_dim, base_length_scale, dtype=np.float32)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=length_scales)
        self.model = GaussianProcessClassifier(kernel=kernel, warm_start=True, random_state=0)
        self._is_fitted = False
        self._single_class_probability: Optional[float] = None

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Return P(class=1) for a single feature vector. If the model has not
        been fitted yet, return a neutral probability (0.5).
        """
        if self._single_class_probability is not None:
            return self._single_class_probability
        if not self._is_fitted:
            return 0.5
        feature_vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
        probabilities = self.model.predict_proba(feature_vector)
        return float(probabilities[0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Optional[float]:
        """
        Fit the GP classifier on all available samples.
        Returns a log-loss value for logging purposes.
        """
        if X is None or len(X) == 0:
            return None

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32).ravel()
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {X.shape[1]}")

        unique_classes = np.unique(y)
        if unique_classes.size < 2:
            # Not enough diversity for GP fit; treat as deterministic probability.
            label_value = float(unique_classes[0])
            self._single_class_probability = label_value
            self._is_fitted = False
            return 0.0

        self._single_class_probability = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            self.model.fit(X, y)

        self._is_fitted = True
        probs = np.clip(self.model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return float(log_loss(y, probs))
