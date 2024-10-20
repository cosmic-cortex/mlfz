import pytest
import numpy as np
from mlfz.classical.linear import *


def generate_linear_dataset(n_samples, n_features):
    X = np.random.rand(n_samples, n_features)
    a = np.random.rand(n_features)
    b = np.random.rand()
    Y = X @ a + b
    return {"X": X, "Y": Y, "a": a, "b": b}


@pytest.fixture
def datasets():
    dims = [1, 2, 5]
    return [generate_linear_dataset(1000, d) for d in dims]


def test_linear_gd(datasets):
    for d in datasets:
        X, Y, a, b = d["X"], d["Y"], d["a"], d["b"]
        n_samples, n_features = X.shape

        linear = LinearRegressorGD(n_features=n_features)

        # testing .fit and correctness
        linear.fit(X, Y, lr=0.1, n_steps=10000)
        assert np.allclose(linear.a, a, rtol=1e-3)
        assert np.allclose(linear.b, b, rtol=1e-3)

        # testing the other methods
        linear.fit_predict(X, Y)
        linear.predict(X)
        linear(X)


def test_linear_gd_v2(datasets):
    for d in datasets:
        X, Y, a, b = d["X"], d["Y"], d["a"], d["b"]
        w = np.concatenate((a, np.array([b])))
        n_samples, n_features = X.shape

        linear = LinearRegressorGDV2(n_features=n_features)

        # testing .fit and correctness
        linear.fit(X, Y, lr=0.1, n_steps=10000)
        assert np.allclose(linear.w, w, rtol=1e-3)

        # testing the other methods
        linear.fit_predict(X, Y)
        linear.predict(X)
        linear(X)


def test_linear_ne(datasets):
    for d in datasets:
        X, Y, a, b = d["X"], d["Y"], d["a"], d["b"]
        n_samples, n_features = X.shape

        linear = LinearRegressorLS(n_features=n_features)

        # testing .fit and correctness
        linear.fit(X, Y)
        assert np.allclose(linear.w[:-1], a, rtol=1e-3)
        assert np.allclose(linear.w[-1], b, rtol=1e-3)

        # testing the other methods
        linear.fit_predict(X, Y)
        linear.predict(X)
        linear(X)
