import numpy as np
from .base import Model


class Normalize(Model):
    """
    Normalizes the features by removing the mean and standard deviation.
    """

    def fit(self, X: np.array, *args):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing
                the training data.
        """

        self.sample_std = X.std(axis=0)
        self.sample_mean = X.mean(axis=0)

    def predict(self, X: np.array, *args):
        return (X - self.sample_mean) / self.sample_std

    def decode(self, X: np.array):
        return X * self.sample_std + self.sample_mean
