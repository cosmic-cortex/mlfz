import numpy as np
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


def test_sin():
    f = lambda x: sin(x).sum()
    f_np = lambda x: np.sin(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_cos():
    f = lambda x: cos(x).sum()
    f_np = lambda x: np.cos(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_exp():
    f = lambda x: exp(x).sum()
    f_np = lambda x: np.exp(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_log():
    f = lambda x: log(x).sum()
    f_np = lambda x: np.log(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_sigmoid():
    f = lambda x: sigmoid(x).sum()
    f_np = lambda x: _sigmoid(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_tanh():
    f = lambda x: tanh(x).sum()
    f_np = lambda x: np.tanh(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )


def test_relu():
    f = lambda x: relu(x).sum()
    f_np = lambda x: _relu(x).sum()

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    x_1d = Tensor.from_random(5)
    x_2d = Tensor.from_random(5, 6)
    x_3d = Tensor.from_random(5, 6, 7)

    for x in [x_1d]:
        y = f(x)
        y.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(f_np, x.value),
        )
