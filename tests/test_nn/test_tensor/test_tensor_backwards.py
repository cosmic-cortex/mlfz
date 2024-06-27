import numpy as np
from mlfz.nn.tensor import Tensor
from functools import partial


def _finite_diff(f, x, eps=1e-8):
    return (f(x + eps) - f(x)) / eps


def test_add():
    # tensor case
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))

    f = lambda x, y: x + y
    z = x + y
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y), x))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x), y))


def test_sub_pow():
    # tensor case
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))

    f = lambda x, y: (x - y) ** 3
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y), x))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x), y))
