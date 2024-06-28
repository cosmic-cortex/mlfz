import numpy as np
from mlfz.nn.tensor import Tensor
from functools import partial


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
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    y = Tensor(np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]))

    f = lambda x, y: (x + y).sum()
    z = f(x, y)
    z.backward()

    assert np.allclose(x.backwards_grad, _finite_diff(partial(f, y=y.value), x.value))
    assert np.allclose(y.backwards_grad, _finite_diff(partial(f, x.value), y.value))


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


def test_reshape_sum():
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
