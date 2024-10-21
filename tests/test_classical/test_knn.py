import pytest
import numpy as np
from mlfz.classical.knn import *
from mlfz.datasets import *


@pytest.fixture
def ds_classification():
    return [
        generate_clusters(2, 1, 100, spread=0.1),
        generate_clusters(2, 2, 100, spread=0.1),
        generate_clusters(2, 5, 100, spread=0.1),
        generate_clusters(5, 1, 100, spread=0.1),
        generate_clusters(5, 2, 100, spread=0.1),
        generate_clusters(5, 5, 100, spread=0.1),
    ]


@pytest.fixture
def ds_regression():
    dims = [1, 2, 5]
    return [generate_linear_regression_ds(1000, d) for d in dims]


def test_knn_classifier(ds_classification):
    for d in ds_classification:
        X, Y = d["X"], d["Y"]

        for k in [1, 2, 5]:
            knn = KNNClassifier(k)

            knn.fit(X, Y)
            Y_pred = knn.predict(X)
            assert np.all(Y_pred == Y)

            knn.fit_predict(X, Y)
            knn(X)


def test_knn_regressor(ds_regression):
    for d in ds_regression:
        X, Y = d["X"], d["Y"]

        for k in [1, 2, 5]:
            knn = KNNRegressor(k)

            knn.fit(X, Y)
            preds = knn.fit_predict(X, Y)
            assert preds.shape == Y.shape
            preds = knn.predict(X)
            assert preds.shape == Y.shape
            knn(X)
