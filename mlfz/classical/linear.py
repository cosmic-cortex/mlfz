import numpy as np
from .base import Model
from mlfz.functional.numpy import sigmoid


def _augment_data(X: np.array):
    """
    Augments the data matrix with a column of ones to eliminate the bias from
    the weights.

    Args:
        X: numpy.ndarray of shape (n_batch, n_features) containing
            the input value.

    Returns:
        X_: numpy.ndarray of shape (n_batch, n_features + 1) containing
            the input value.
    """
    ones_column = np.ones((X.shape[0], 1))
    return np.hstack((X, ones_column))


class LinearRegressorGD(Model):
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
            X: numpy.ndarray of shape (n_batch, n_features) containing
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
        Fits the linear regressor with gradient descent.

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


class LinearRegressorGDV2(Model):
    def __init__(self, n_features: int):
        """
        Linear regression, with gradient descent. This implementation stores the
        weights in a single (n_features + 1, ) shaped array.

        Args:
            n_features: the number of features
        """
        self.w = np.zeros(n_features + 1)

    def _grad_L(self, X: np.array, Y: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the corresponding ground truth.

        Returns:
            dw: numpy.ndarray of shape (n_features + 1, ) containing
                the gradient with respect to w.
        """

        n = len(X)
        pred_error = self(X) - Y

        X_ = _augment_data(X)
        dw = (2 / n) * pred_error @ X_

        return dw

    def predict(self, X: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

        Returns:
            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the model predictions.
        """

        X_ = _augment_data(X)
        return X_ @ self.w

    def fit(self, X: np.array, Y: np.array, lr: int = 0.01, n_steps: int = 1000):
        """
        Fits the linear regressor with gradient descent.

        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the corresponding ground truth.

            lr: float describing the learning rate.

            n_steps: integer, describing the number of steps taken by
                the gradient descent.
        """

        for _ in range(n_steps):
            dw = self._grad_L(X, Y)
            self.w = self.w - lr * dw


class LinearRegressorLS(Model):
    def __init__(self, n_features: int):
        """
        Linear regression, optimized by solving the normal equations.

        Args:
            n_features: the number of features
        """
        self.w = np.zeros(n_features + 1)

    def predict(self, X: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

        Returns:
            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the model predictions.
        """

        X_ = _augment_data(X)
        return X_ @ self.w

    def fit(self, X: np.array, Y: np.array):
        """
        Fits the linear regressor by solving the normal equations.

        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
                the input value.

            Y: numpy.ndarray of shape of shape (n_batch, ) containing
                the corresponding ground truth.
        """

        X_ = _augment_data(X)
        self.w = np.linalg.inv(X_.T @ X_) @ X_.T @ Y


class BinaryLogistic(Model):
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

        return 1 * (sigmoid(X @ self.a + self.b) > 0.5)

    def _grad_L(self, X: np.array, Y: np.array):
        """
        Args:
            X: numpy.ndarray of shape (n_batch, n_features) containing
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

        da = (1 / n) * pred_error @ X
        db = (1 / n) * pred_error.sum()
        return da, db

    def fit(self, X: np.array, Y: np.array, lr: int = 0.01, n_steps: int = 1000):
        """
        Fits the linear regressor with gradient descent.

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
