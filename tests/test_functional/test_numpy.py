import numpy as np
from mlfz.functional import numpy as f


def test_output_type():
    for func in [
        f.sigmoid,
        f.sigmoid_prime,
        f.relu,
        f.relu_prime,
        f.tanh,
        f.tanh_prime,
    ]:
        X = np.random.rand(5, 3)
        assert isinstance(func(X), np.ndarray)
