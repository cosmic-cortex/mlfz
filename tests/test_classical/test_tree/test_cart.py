import pytest
import numpy as np
from mlfz.datasets import generate_clusters
from mlfz.classical.tree.cart import *


@pytest.fixture
def datasets():
    return [
        generate_clusters(1, 1, 10, spread=0.1),
        generate_clusters(1, 2, 10, spread=0.1),
        generate_clusters(1, 5, 10, spread=0.1),
        generate_clusters(2, 1, 10, spread=0.1),
        generate_clusters(2, 2, 10, spread=0.1),
        generate_clusters(2, 5, 10, spread=0.1),
        generate_clusters(5, 1, 10, spread=0.1),
        generate_clusters(5, 2, 10, spread=0.1),
        generate_clusters(5, 5, 10, spread=0.1),
    ]


def test_classification_tree(datasets):
    for d in datasets:
        for max_depth in [0, 1, 2, 5]:
            X, Y = d["X"], d["Y"]

            tree = ClassificationTree(max_depth=max_depth)

            tree.fit(X, Y)
            tree.fit_predict(X, Y)
            tree.predict(X)
            tree(X)
