import pytest
import numpy as np
from mlfz.classical.knn import *
from mlfz.datasets import generate_clusters


@pytest.fixture
def datasets():
    return [
        generate_clusters(2, 1, 100),
        generate_clusters(2, 2, 100),
        generate_clusters(2, 5, 100),
        generate_clusters(5, 1, 100),
        generate_clusters(5, 2, 100),
        generate_clusters(5, 5, 100),
    ]


def test_knn_classifier(datasets):
    for d in datasets:
        X, Y = d["X"], d["Y"]

        for k in [1, 2, 5]:
            knn = KNNClassifier(k)

            knn.fit(X, Y)
            Y_pred = knn.predict(X)
            # assert np.all(Y_pred == Y)

            knn.fit_predict(X, Y)
            knn(X)
