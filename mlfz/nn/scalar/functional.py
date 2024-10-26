import math
from mlfz.nn.scalar.core import Scalar, Edge
from mlfz.functional import vanilla as f


def sin(x: Scalar):
    return Scalar(value=math.sin(x.value), prevs=[Edge(x, math.cos(x.value))])


def cos(x: Scalar):
    return Scalar(value=math.cos(x.value), prevs=[Edge(x, -math.sin(x.value))])


def exp(x: Scalar):
    return Scalar(value=math.exp(x.value), prevs=[Edge(x, math.exp(x.value))])


def log(x: Scalar):
    return Scalar(value=math.log(x.value), prevs=[Edge(x, 1 / math.fabs(x.value))])


def sigmoid(x: Scalar):
    return Scalar(
        value=f.sigmoid(x.value),
        prevs=[Edge(x, f.sigmoid_prime(x.value))],
    )


def relu(x: Scalar):
    return Scalar(value=f.relu(x.value), prevs=[Edge(x, f.relu_prime(x.value))])


def tanh(x: Scalar):
    return Scalar(value=f.tanh(x.value), prevs=[Edge(x, f.tanh_prime(x.value))])
