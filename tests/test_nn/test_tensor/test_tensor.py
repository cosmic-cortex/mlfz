import numpy as np
from mlfz.nn.tensor import Tensor, sum


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


def test_add():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    # scalar case
    y1 = Tensor(2) + x
    y1_left = x + Tensor(2)
    assert (y1.value == 2 + x.value).all()
    assert (y1_left.value == x.value + 2).all()

    # tensor case
    t_1d_col = Tensor(np.array([[1], [2], [3]]))
    t_1d_row = Tensor(np.array([1, 2]))
    t_2d = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y2 = t_1d_col + x
    y2_left = x + t_1d_col
    y3 = x + t_1d_row
    y3_left = t_1d_row + x
    y4 = t_2d + x
    y4_left = x + t_2d
    assert (
        y2.value == np.array([[1 + 1, 2 + 1], [3 + 2, 4 + 2], [5 + 3, 6 + 3]])
    ).all()
    assert (
        y2_left.value == np.array([[1 + 1, 2 + 1], [3 + 2, 4 + 2], [5 + 3, 6 + 3]])
    ).all()
    assert (
        y3.value == np.array([[1 + 1, 2 + 2], [3 + 1, 4 + 2], [5 + 1, 6 + 2]])
    ).all()
    assert (
        y3_left.value == np.array([[1 + 1, 2 + 2], [3 + 1, 4 + 2], [5 + 1, 6 + 2]])
    ).all()
    assert (
        y4.value == np.array([[1 + 1, 2 + 2], [3 + 3, 4 + 4], [5 + 5, 6 + 6]])
    ).all()
    assert (
        y4_left.value == np.array([[1 + 1, 2 + 2], [3 + 3, 4 + 4], [5 + 5, 6 + 6]])
    ).all()


def test_sum():
    x = Tensor(np.array([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))
    axs = [None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    for axis in axs:
        x_sum = sum(x, axis=axis)
        x_sum_method = x.sum(axis=axis)
        assert (x_sum.value == np.sum(x.value, axis=axis)).all()
        assert (x_sum.value == x_sum_method.value).all()


def test_reshape():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    assert (x.reshape(2, 3).value == x.value.reshape(2, 3)).all()
    assert (x.reshape(1, -1).value == x.value.reshape(1, -1)).all()
    assert (x.reshape(-1).value == x.value.reshape(-1)).all()


def test_mul():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    # scalar case
    y1 = Tensor(2) * x
    y1_left = x * Tensor(2)
    assert (y1.value == 2 * x.value).all()
    assert (y1_left.value == x.value * 2).all()

    # tensor case
    t_1d_col = Tensor(np.array([[1], [2], [3]]))
    t_1d_row = Tensor(np.array([1, 2]))
    t_2d = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y2 = t_1d_col * x
    y2_left = x * t_1d_col
    y3 = x * t_1d_row
    y3_left = t_1d_row * x
    y4 = t_2d * x
    y4_left = x * t_2d
    assert (
        y2.value == np.array([[1 * 1, 2 * 1], [3 * 2, 4 * 2], [5 * 3, 6 * 3]])
    ).all()
    assert (
        y2_left.value == np.array([[1 * 1, 2 * 1], [3 * 2, 4 * 2], [5 * 3, 6 * 3]])
    ).all()
    assert (
        y3.value == np.array([[1 * 1, 2 * 2], [3 * 1, 4 * 2], [5 * 1, 6 * 2]])
    ).all()
    assert (
        y3_left.value == np.array([[1 * 1, 2 * 2], [3 * 1, 4 * 2], [5 * 1, 6 * 2]])
    ).all()
    assert (
        y4.value == np.array([[1 * 1, 2 * 2], [3 * 3, 4 * 4], [5 * 5, 6 * 6]])
    ).all()
    assert (
        y4_left.value == np.array([[1 * 1, 2 * 2], [3 * 3, 4 * 4], [5 * 5, 6 * 6]])
    ).all()


def test_div():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    # scalar case
    y1 = x / Tensor(2)
    y1_left = Tensor(2) / x
    assert (y1.value == x.value / 2).all()
    assert (y1_left.value == 2 / x.value).all()

    # tensor case
    t_1d_col = Tensor(np.array([[1], [2], [3]]))
    t_1d_row = Tensor(np.array([1, 2]))
    t_2d = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y2 = x / t_1d_col
    y3 = x / t_1d_row
    y4 = x / t_2d
    assert (
        y2.value == np.array([[1 / 1, 2 / 1], [3 / 2, 4 / 2], [5 / 3, 6 / 3]])
    ).all()
    assert (
        y3.value == np.array([[1 / 1, 2 / 2], [3 / 1, 4 / 2], [5 / 1, 6 / 2]])
    ).all()
    assert (
        y4.value == np.array([[1 / 1, 2 / 2], [3 / 3, 4 / 4], [5 / 5, 6 / 6]])
    ).all()


def test_pow():
    x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))

    # scalar case
    y1 = x**2
    y2 = 2**x
    assert (y1.value == np.array([[1, 2**2], [3**2, 4**2], [5**2, 6**2]])).all()
    assert (y2.value == np.array([[2**1, 2**2], [2**3, 2**4], [2**5, 2**6]])).all()

    # tensor case
    t_1d_col = Tensor(np.array([[1], [2], [3]]))
    t_1d_row = Tensor(np.array([1, 2]))
    t_2d = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y3 = x**t_1d_col
    y4 = x**t_1d_row
    y5 = x**t_2d
    assert (y3.value == np.array([[1**1, 2**1], [3**2, 4**2], [5**3, 6**3]])).all()
    assert (y4.value == np.array([[1**1, 2**2], [3**1, 4**2], [5**1, 6**2]])).all()
    assert (y5.value == np.array([[1**1, 2**2], [3**3, 4**4], [5**5, 6**6]])).all()


def test_broadcast_to():
    x = Tensor.ones(3, 1)
    shapes = [(3, 2), (3, 5), (3, 9)]
    for s in shapes:
        assert x.broadcast_to(s).shape == s

    x = Tensor(2)
    shapes = [(1, 1, 1), (3, 1), (1, 3), (1, 2, 1), (3, 5, 6)]
    for s in shapes:
        assert x.broadcast_to(s).shape == s
