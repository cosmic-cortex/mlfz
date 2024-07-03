import numpy as np
from mlfz.nn.tensor import Tensor, sum, mean
from itertools import product


def test_init():
    x_shape = (5, 4)
    x_ones_true = np.ones(x_shape)
    x_zeros_true = np.zeros(x_shape)

    x = Tensor.from_random(*x_shape)
    assert x.shape == x_shape
    assert x[0, 0] == x.value[0, 0]

    x_ones = Tensor.ones(*x_shape)
    assert (x_ones.value == x_ones_true).all()

    x_ones_like = Tensor.ones_like(x)
    assert x_ones_like.shape == x_shape
    assert (x_ones_like.value == x_ones_true).all()

    x_ones_like_from_np = Tensor.ones_like(x.value)
    assert x_ones_like_from_np.shape == x.shape
    assert (x_ones_like_from_np.value == x_ones_true).all()

    x_zeros = Tensor.zeros(*x_shape)
    assert (x_zeros.value == x_zeros_true).all()

    x_zeros_like = Tensor.zeros_like(x)
    assert x_zeros_like.shape == x_shape
    assert (x_zeros_like.value == x_zeros_true).all()

    x_zeros_like_from_np = Tensor.zeros_like(x.value)
    assert x_zeros_like_from_np.shape == x.shape
    assert (x_zeros_like_from_np.value == x_zeros_true).all()


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
        z_np = f(x.value, y.value)
        assert (z.value == z_np).all()


def test_sum():
    x = Tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))
    axs = [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axis in axs:
        x_sum = sum(x, axis=axis)
        x_sum_method = x.sum(axis=axis)
        assert (x_sum.value == np.sum(x.value, axis=axis)).all()
        assert (x_sum.value == x_sum_method.value).all()


def test_mean():
    x = Tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))
    axs = [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axis in axs:
        x_mean = mean(x, axis=axis)
        x_mean_method = x.mean(axis=axis)
        assert (x_mean.value == np.mean(x.value, axis=axis)).all()
        assert (x_mean.value == x_mean_method.value).all()


def test_reshape():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    assert (x.reshape(2, 3).value == x.value.reshape(2, 3)).all()
    assert (x.reshape(1, -1).value == x.value.reshape(1, -1)).all()
    assert (x.reshape(-1).value == x.value.reshape(-1)).all()


def test_broadcast_to():
    x = Tensor.ones(3, 1)
    shapes = [(3, 2), (3, 5), (3, 9)]
    for s in shapes:
        assert x.broadcast_to(s).shape == s

    x = Tensor(2)
    shapes = [(1, 1, 1), (3, 1), (1, 3), (1, 2, 1), (3, 5, 6)]
    for s in shapes:
        assert x.broadcast_to(s).shape == s
