import pytest
import numpy as np
from mlfz.classical import *


@pytest.fixture
def datasets():
    return [
        (np.random.rand(10, 1), np.random.rand(10)),
        (np.random.rand(10, 2), np.random.rand(10)),
        (np.random.rand(10, 5), np.random.rand(10)),
    ]


class DummyModel:
    def __init__(self, val):
        self.val = val

    def predict(self, X):
        return X + self.val

    def fit(self, X, Y):
        pass


def test_pipeline(datasets):
    for X, Y in datasets:
        n_samples, n_features = X.shape

        scaler = Normalize()
        linear = LinearRegressorGD(n_features=n_features)
        pipeline = Pipeline([scaler, linear])

        pipeline.fit(X, Y)
        pipeline(X)
        pipeline.predict(X)


def test_pipeline_predict(datasets):
    add_1 = DummyModel(1)
    add_2 = DummyModel(2)
    pipeline = Pipeline([add_1, add_2])

    for X, Y in datasets:
        result = pipeline.predict(X)
        print(X, result)
        assert np.all(result == X + 3)
