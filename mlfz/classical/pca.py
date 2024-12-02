import numpy as np
from .base import Model


class PCA(Model):
    def fit(self, X: np.ndarray):
        X_cov = np.cov(X, rowvar=False)
        U, S, Vt = np.linalg.svd(X_cov)

        sorted_indices = np.argsort(S)[::-1]

        self.S = S[sorted_indices]
        self.U = U[:, sorted_indices]

    def predict(self, X: np.array, *args):
        return X @ self.U
