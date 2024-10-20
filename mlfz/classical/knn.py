import numpy as np
from .base import Model


class KNNClassifier(Model):
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class KNNRegressor(Model):
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass
