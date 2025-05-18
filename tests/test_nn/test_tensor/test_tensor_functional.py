import numpy as np
from itertools import product
from mlfz.nn.tensor import Tensor
from mlfz.nn.tensor.functional import *


def _sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def _relu(x: np.ndarray):
    return x * (x > 0)


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


def test_functional():
    xs = [Tensor.ones(5), Tensor.ones(5, 6), Tensor.ones(5, 6, 7)]
    fs = [
        (lambda x: sin(x).sum(), lambda x: np.sin(x).sum()),
        (lambda x: cos(x).sum(), lambda x: np.cos(x).sum()),
        (lambda x: exp(x).sum(), lambda x: np.exp(x).sum()),
        (lambda x: log(x).sum(), lambda x: np.log(x).sum()),
        (lambda x: sigmoid(x).sum(), lambda x: _sigmoid(x).sum()),
        (lambda x: tanh(x).sum(), lambda x: np.tanh(x).sum()),
        (lambda x: relu(x).sum(), lambda x: _relu(x).sum()),
    ]

    for x, f in product(xs, fs):
        f_tensor, f_np = f
        y = f_tensor(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_pad():
    xs = [Tensor.ones(5), Tensor.ones(5, 6)]  # , Tensor.ones(5, 6, 7)]
    ws = [0, 1, 2, 3]
    cs = [0, 1, 2, 3]

    for x, w, c in zip(xs, ws, cs):
        f = lambda x: pad(x, w, c)
        f_np = lambda x: np.pad(x, pad_width=w, constant_values=c)

        l = lambda x: f(x).sum()
        l_np = lambda x: f_np(x).sum()

        assert (f(x).value == f_np(x.value)).all()

        y = l(x)
        y.backward()

        assert np.allclose(x.backwards_grad, _finite_diff(l_np, x.value))
