from mlfz.nn import Tensor
from mlfz.nn.tensor.functional import *


def test_sin():
    x_1d = Tensor.from_random(5)
    sin(x_1d)

    x_2d = Tensor.from_random(5, 6)
    sin(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    sin(x_3d)


def test_cos():
    x_1d = Tensor.from_random(5)
    cos(x_1d)

    x_2d = Tensor.from_random(5, 6)
    cos(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    cos(x_3d)


def test_exp():
    x_1d = Tensor.from_random(5)
    exp(x_1d)

    x_2d = Tensor.from_random(5, 6)
    exp(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    exp(x_3d)


def test_log():
    x_1d = Tensor.from_random(5)
    log(x_1d)

    x_2d = Tensor.from_random(5, 6)
    log(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    log(x_3d)


def test_sigmoid():
    x_1d = Tensor.from_random(5)
    sigmoid(x_1d)

    x_2d = Tensor.from_random(5, 6)
    sigmoid(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    sigmoid(x_3d)


def test_tanh():
    x_1d = Tensor.from_random(5)
    tanh(x_1d)

    x_2d = Tensor.from_random(5, 6)
    tanh(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    tanh(x_3d)


def test_relu():
    x_1d = Tensor.from_random(5)
    relu(x_1d)

    x_2d = Tensor.from_random(5, 6)
    relu(x_2d)

    x_3d = Tensor.from_random(5, 6, 7)
    relu(x_3d)
