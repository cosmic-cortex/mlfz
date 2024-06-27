import numpy as np
from mlfz.nn.tensor import Tensor
from functools import partial


# def _finite_diff(f, x, eps=1e-8):
#     return (f(x + eps) - f(x)) / eps


def _finite_diff(f, x, h=1e-8):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[idx] += h
        x_backward[idx] -= h
        print(x_forward.dtype, x_backward)
        grad[idx] = (f(x_forward) - f(x_backward)) / (2 * h)

    return grad.reshape(x.shape)


def test_add():
    # tensor case
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))

    f = lambda x, y: (x + y).sum()
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y.value), x.value))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x.value), y.value))


def test_sub_pow():
    # tensor case
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y = Tensor(np.array([[7, 8], [9, 10], [11, 12]]))

    f = lambda x, y: ((x - y) ** 3).sum()
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y.value), x.value))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x.value), y.value))


def test_reshape_sum():
    x = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]))

    fs = [
        lambda x: x.reshape(1, -1).sum(),
        lambda x: x.reshape(-1).sum(),
        lambda x: x.reshape(2, 6).sum(),
        lambda x: x.reshape(6, 2).sum(),
        lambda x: x.reshape(4, 3).sum(),
    ]

    for f in fs:
        y = f(x)
        y.backward()
        assert np.allclose(y.backwards_grad, _finite_diff(f, x.value))
