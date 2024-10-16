import numpy as np
from mlfz.classical.linear import *


def test_linear():
    datasets = [
        (np.random.rand(10, 1), np.random.rand(10)),
        (np.random.rand(10, 2), np.random.rand(10)),
        (np.random.rand(10, 5), np.random.rand(10)),
    ]

    for X, Y in datasets:
        n_samples, n_features = X.shape

        linear = MultivariateLinearRegressorGD(n_features=n_features)
        linear.fit(X, Y)
        linear.fit_predict(X, Y)
        linear.predict(X)
        linear(X)
