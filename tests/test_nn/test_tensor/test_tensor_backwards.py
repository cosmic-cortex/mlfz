import numpy as np
from mlfz.nn.tensor import Tensor
from functools import partial
from itertools import product


def _finite_diff(f, x, h=1e-8):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[idx] += h
        x_backward[idx] -= h
        grad[idx] = (f(x_forward) - f(x_backward)) / (2 * h)

    return grad.reshape(x.shape)


def test_binary_ops():
    x = 2 * Tensor.ones(3, 2)
    ys = [
        Tensor(2),
        2 * Tensor.ones(3, 2),
        2 * Tensor.ones(3, 1),
        2 * Tensor.ones(1, 2),
    ]

    fs = [
        lambda x, y: (x + y).sum(),
        lambda x, y: (y + x).sum(),
        lambda x, y: (x * y).sum(),
        lambda x, y: (y * x).sum(),
        lambda x, y: (x**y).sum(),
        lambda x, y: (x / y).sum(),
    ]

    for f, y in product(fs, ys):
        z = f(x, y)
        z.backward()
        assert np.allclose(
            x.backwards_grad,
            _finite_diff(partial(f, y=y.value), x.value),
            1e-4,
        )
        assert np.allclose(
            y.backwards_grad,
            _finite_diff(partial(f, x.value), y.value),
            1e-4,
        )


def test_sum():
    x = Tensor(
        np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
    )
    axs = [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axis in axs:
        f = lambda x: x.sum(axis=axis).sum()
        y = f(x)
        y.backward()
        assert np.allclose(x.backwards_grad, _finite_diff(f, x.value))


def test_sub_pow():
    # tensor case
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    y = Tensor(np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))

    f = lambda x, y: ((x - y) ** 3).sum()
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y.value), x.value))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x.value), y.value))


def test_reshape():
    x = Tensor(
        np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    )

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


def test_broadcast_to():
    x = Tensor.ones(3, 1)
    shapes = [(3, 2), (3, 5), (3, 9)]

    for s in shapes:
        f = lambda x: x.broadcast_to(s).sum()
        f_np = lambda x: np.broadcast_to(x, s).sum()

        y = f(x)
        y.backward()

        assert x.backwards_grad.shape == x.shape
        assert np.allclose(x.backwards_grad, _finite_diff(f_np, x.value))

    x = Tensor.ones(1, 3)
    shapes = [(2, 3), (5, 3), (9, 3)]

    for s in shapes:
        f = lambda x: x.broadcast_to(s).sum()
        f_np = lambda x: np.broadcast_to(x, s).sum()

        y = f(x)
        y.backward()

        assert x.backwards_grad.shape == x.shape
        assert np.allclose(x.backwards_grad, _finite_diff(f_np, x.value))


def test_matmul():
    x = Tensor.ones(5, 4)
    y = Tensor.ones(4, 8)

    f = lambda x, y: (x @ y).sum()
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y.value), x.value))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x.value), y.value))


def test_transpose():
    x = Tensor.ones(5, 8)
    f = lambda x: x.T.sum()
    y = f(x)
    y.backward()

    assert np.allclose(y.backwards_grad, _finite_diff(f, x.value))
