from mlfz.nn import Scalar
from mlfz.nn.scalar.functional import *


def test_sin():
    x = Scalar.from_random()
    sin(x)


def test_cos():
    x = Scalar.from_random()
    cos(x)


def test_exp():
    x = Scalar.from_random()
    exp(x)


def test_log():
    x = Scalar.from_random()
    log(x)


def test_sigmoid():
    x = Scalar.from_random()
    sigmoid(x)


def test_tanh():
    x = Scalar.from_random()
    tanh(x)


def test_relu():
    x = Scalar.from_random()
    relu(x)
