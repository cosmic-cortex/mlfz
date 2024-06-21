import numpy as np
from mlfz.nn.tensor import Tensor


def test_init():
    x_shape = (5, 4)
    x = Tensor.from_random(*x_shape)
    assert x.shape == x_shape

    x_ones = Tensor.ones_like(x)
    assert x_ones.shape == x_shape

    x_zeros = Tensor.zeros_like(x)
    assert x_zeros.shape == x_shape

    x_ones_from_np = Tensor.ones_like(x.value)
    assert x_ones_from_np.shape == x.shape

    x_zeros_from_np = Tensor.zeros_like(x.value)
    assert x_zeros_from_np.shape == x.shape


def test_add():
    x, y = Tensor.from_random(5, 8), Tensor.from_random(5, 8)
    z = x + y
    assert (z.value == x.value + y.value).all()
