import numpy as np
from abc import ABC, abstractmethod


class Model:
    def __call__(self, *args, **kwargs):
        return self.predict(*args, *kwargs)

    @abstractmethod
    def fit(self, X: np.array, Y: np.array):
        pass

    @abstractmethod
    def predict(self, X: np.array):
        pass

    def fit_predict(self, X: np.array, Y: np.array):
        self.fit(X, Y)
        return self.predict(X)
