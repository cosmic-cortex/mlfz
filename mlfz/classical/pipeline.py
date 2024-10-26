import numpy as np
from typing import Iterable
from .base import Model


class Pipeline(Model):
    def __init__(self, components: Iterable):
        self.components = components

    def predict(self, X: np.array):
        for f in self.components:
            X = f.predict(X)

        return X

    def fit(self, X: np.array, Y: np.array):
        for f in self.components:
            X = f.fit_predict(X, Y)
