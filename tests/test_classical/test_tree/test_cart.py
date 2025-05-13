import pytest
import numpy as np
from mlfz.datasets import generate_clusters, generate_linear_regression_ds
from mlfz.classical.tree.cart import *


@pytest.fixture
def cls_datasets():
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


@pytest.fixture
def rgs_datasets():
    return [
        generate_linear_regression_ds(100, 1),
        generate_linear_regression_ds(100, 2),
        generate_linear_regression_ds(100, 5),
    ]


def test_classification_tree(cls_datasets):
    for d in cls_datasets:
        X, Y = d["X"], d["Y"]
        for max_depth in [0, 1, 2, 5]:
            tree = ClassificationTree(max_depth=max_depth)
            tree.fit(X, Y)
            tree.fit_predict(X, Y)
            tree.predict(X)
            tree(X)


def test_regression_tree(rgs_datasets):
    for d in rgs_datasets:
        X, Y = d["X"], d["Y"]
        for max_depth in [0, 1, 2, 5]:
            tree = RegressionTree(max_depth=max_depth)
            tree.fit(X, Y)
            tree.fit_predict(X, Y)
            tree.predict(X)
            tree(X)
