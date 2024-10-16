import numpy as np
from mlfz.classical.scaler import *


def test_normalize():
    datasets = [np.random.rand(10), np.random.rand(10, 11), np.random.rand(10, 11, 12)]

    for X in datasets:
        scaler = Normalize()
        scaler.fit(X)
        scaler.predict(X)
        scaler(X)
        scaler.decode(X)
        scaler.fit_predict(X, None)
