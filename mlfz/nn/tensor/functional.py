import numpy as np
from .core import Tensor, Edge


import numpy as np


def _sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def _sigmoid_prime(x: np.ndarray):
    sigmoid_x = _sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def _relu(x: np.ndarray):
    return x * (x > 0)


def _relu_prime(x: np.ndarray):
    return (x > 0).astype(int)


def _tanh_prime(x: np.ndarray):
    return 1 - np.tanh(x) ** 2


def exp(x: Tensor):
    return Tensor(
        value=np.exp(x.value), prevs=[Edge(prev=x, local_grad=np.exp(x.value))]
    )


def log(x: Tensor):
    return Tensor(value=np.log(x.value), prevs=[Edge(prev=x, local_grad=1 / np.abs(x))])


def sin(x: Tensor):
    return Tensor(
        value=np.sin(x.value), prevs=[Edge(prev=x, local_grad=np.cos(x.value))]
    )


def cos(x: Tensor):
    return Tensor(
        value=np.cos(x.value), prevs=[Edge(prev=x, local_grad=-np.sin(x.value))]
    )


def sigmoid(x: Tensor):
    return Tensor(
        value=_sigmoid(x.value),
        prevs=[Edge(prev=x, local_grad=_sigmoid_prime(x.value))],
    )


def relu(x: Tensor):
    return Tensor(
        value=_relu(x.value),
        prevs=[Edge(prev=x, local_grad=_relu_prime(x.value))],
    )


def tanh(x: Tensor):
    return Tensor(
        value=np.tanh(x.value),
        prevs=[Edge(prev=x, local_grad=_tanh_prime(x.value))],
    )
