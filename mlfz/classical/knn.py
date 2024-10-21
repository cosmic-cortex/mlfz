import numpy as np
from scipy.spatial.distance import cdist
from .base import Model


class KNNClassifier(Model):
    def __init__(self, k: int, metric: str = "euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, Y):
        self.X_training = X
        self.Y_training = Y

    def predict(self, X):
        dmap = cdist(self.X_training, X)
        nearest_idx = np.argsort(dmap, axis=0)[: self.k, :]
        nearest_lbl = self.Y_training[nearest_idx]
        pred = np.array([np.bincount(labels).argmax() for labels in nearest_lbl.T])
        return pred


class KNNRegressor(Model):
    def __init__(self, k: int, metric: str = "euclidean"):
        self.k = k
        self.metric = metric

    def fit(self, X, Y):
        self.X_training = X
        self.Y_training = Y

    def predict(self, X):
        dmap = cdist(self.X_training, X)
        nearest_idx = np.argsort(dmap, axis=0)[: self.k, :]
        nearest_lbl = self.Y_training[nearest_idx]
        pred = nearest_lbl.mean(axis=0)
        return pred
