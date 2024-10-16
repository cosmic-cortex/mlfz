import numpy as np
from mlfz.classical.linear import *


def test_Linear():
    datasets = [
        (np.random.rand(5, 4), np.random.rand(5)),
        (np.random.rand(24, 8), np.random.rand(24)),
    ]

    for X, Y in datasets:
        n_samples, n_features = X.shape
        linear = MultivariateLinearRegressorGD(n_features=n_features)
        linear.fit(X, Y)
        linear.predict(X)
