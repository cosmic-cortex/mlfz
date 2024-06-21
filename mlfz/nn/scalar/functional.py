import math
from .core import Scalar, Edge


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _sigmoid_prime(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


def _relu(x):
    return max([0, x])


def _relu_prime(x):
    if x < 0:
        return 0
    else:
        return 1


def _log_prime(x):
    if x != 0:
        return 1 / math.fabs(x)
    else:
        return 0


def _tanh_prime(x):
    return 1 - math.tanh(x) ** 2


def sin(x: Scalar):
    return Scalar(value=math.sin(x.value), prevs=[Edge(x, math.cos(x.value))])


def cos(x: Scalar):
    return Scalar(value=math.cos(x.value), prevs=[Edge(x, -math.sin(x.value))])


def exp(x: Scalar):
    return Scalar(value=math.exp(x.value), prevs=[Edge(x, math.exp(x.value))])


def log(x: Scalar):
    return Scalar(value=math.log(x.value), prevs=[Edge(x, _log_prime(x.value))])


def sigmoid(x: Scalar):
    return Scalar(
        value=_sigmoid(x.value),
        prevs=[Edge(x, _sigmoid_prime(x.value))],
    )


def relu(x: Scalar):
    return Scalar(value=_relu(x.value), prevs=[Edge(x, _relu_prime(x.value))])


def tanh(x: Scalar):
    return Scalar(value=math.tanh(x.value), prevs=[Edge(x, _tanh_prime(x.value))])
