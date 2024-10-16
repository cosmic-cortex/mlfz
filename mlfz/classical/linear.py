import numpy as np
from .base import Model


class MultivariateLinearRegressorGD(Model):
    def __init__(self, n_features: int):
        """
        Linear regression, optimized with gradient descent.

        Args:
            n_features: the number of features
        """
        self.a = np.zeros(n_features)
        self.b = 0

    def predict(self, X: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

        Returns:
            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the model predictions.
        """

        return X @ self.a + self.b

    def _grad_L(self, X: np.array, Y: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing
                the input value.

            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the corresponding ground truth.

        Returns:
            da: numpy.ndarray of shape (n_features, ) containing
                the gradient with respect to a.

            db: float, containing the gradient with respect to b.
        """

        n = len(X)
        pred_error = self(X) - Y

        da = (2 / n) * pred_error @ X
        db = (2 / n) * pred_error.sum()
        return da, db

    def fit(self, X: np.array, Y: np.array, lr: int = 0.01, n_steps: int = 1000):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing
                the input value.

            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the corresponding ground truth.

            lr: float describing the learning rate.

            n_steps: integer, describing the number of steps taken by
                the gradient descent.
        """

        for _ in range(n_steps):
            da, db = self._grad_L(X, Y)
            self.a, self.b = self.a - lr * da, self.b - lr * db


class MultivariateLinearRegressorLS(Model):
    pass
