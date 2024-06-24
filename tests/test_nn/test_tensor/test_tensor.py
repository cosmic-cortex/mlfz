import numpy as np
from mlfz.nn.tensor import Tensor


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
    x = Tensor.from_random(5, 8)
    y_tensor = Tensor.from_random(5, 8)
    z1 = x + y_tensor
    assert (z1.value == x.value + y_tensor.value).all()

    # test adding NumPy arrays
    y_numpy = np.random.rand(5, 8)
    z2 = x + y_numpy
    z3 = y_numpy + x
    assert (z2.value == x.value + y_numpy).all()
    assert (z3.value == y_numpy + x.value).all()

    # test adding floats
    y_float = 4.32
    z4 = x + y_float
    z5 = y_float + x
    assert (z4.value == x.value + y_float).all()
    assert (z5.value == y_float + x.value).all()


def test_mul():
    x = Tensor.ones(3, 2)

    # scalar case
    y1 = Tensor(2) * x
    y2 = x * Tensor(2)
    assert (y1.value == 2 * x.value).all()
    assert (y2.value == x.value * 2).all()

    # tensor case
    t_1d_col = Tensor(np.array([[1], [2], [3]]))
    t_1d_row = Tensor(np.array([1, 2]))
    t_2d = Tensor(np.array([[1, 2], [3, 4], [5, 6]]))
    y3 = t_1d_col * x
    y4 = x * t_1d_row
    y5 = t_2d * x
    y6 = x * t_2d
    assert (y3.value == np.array([[1, 1], [2, 2], [3, 3]])).all()
    assert (y4.value == np.array([[1, 2], [1, 2], [1, 2]])).all()
    assert (y5.value == np.array([[1, 2], [3, 4], [5, 6]])).all()
    assert (y6.value == np.array([[1, 2], [3, 4], [5, 6]])).all()


def test_div():
    pass


def test_pow():
    x = 2 * Tensor.ones(5, 2)
    y1 = x**2
    assert (y1.value == 4).all()

    y2 = x ** (2 * Tensor.ones(5, 2))
    assert (y2.value == 4).all()
