import pytest
import numpy as np
from mlfz.classical.linear import *
from mlfz.datasets import *


@pytest.fixture
def regression_ds():
    dims = [1, 2, 5]
    return [generate_linear_regression_ds(1000, d) for d in dims]


@pytest.fixture
def classification_ds():
    dims = [1, 2, 5]
    return [generate_linear_classification_ds(1000, d) for d in dims]


def test_linear_gd(regression_ds):
    for d in regression_ds:
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


def test_linear_gd_v2(regression_ds):
    for d in regression_ds:
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


def test_linear_ne(regression_ds):
    for d in regression_ds:
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


def test_binary_logistic(classification_ds):
    for d in classification_ds:
        X, Y, a, b = d["X"], d["Y"], d["a"], d["b"]
        n_samples, n_features = X.shape

        logistic = BinaryLogistic(n_features=n_features)

        logistic.fit(X, Y)
        logistic.fit_predict(X, Y)
        logistic.predict(X)
        logistic(X)
