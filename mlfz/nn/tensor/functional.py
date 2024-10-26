import numpy as np
from mlfz.nn.tensor.core import Tensor, Edge
from mlfz.nn.tensor.core.utils import _pointwise
from mlfz.functional import numpy as f


def exp(x: Tensor):
    return Tensor(
        value=np.exp(x.value),
        prevs=[Edge(prev=x, local_grad=np.exp(x.value), backward_fn=_pointwise)],
    )


def log(x: Tensor):
    return Tensor(
        value=np.log(x.value),
        prevs=[Edge(prev=x, local_grad=1 / np.abs(x.value), backward_fn=_pointwise)],
    )


def sin(x: Tensor):
    return Tensor(
        value=np.sin(x.value),
        prevs=[Edge(prev=x, local_grad=np.cos(x.value), backward_fn=_pointwise)],
    )


def cos(x: Tensor):
    return Tensor(
        value=np.cos(x.value),
        prevs=[Edge(prev=x, local_grad=-np.sin(x.value), backward_fn=_pointwise)],
    )


def sigmoid(x: Tensor):
    return Tensor(
        value=f.sigmoid(x.value),
        prevs=[
            Edge(
                prev=x,
                local_grad=f.sigmoid_prime(x.value),
                backward_fn=_pointwise,
            )
        ],
    )


def relu(x: Tensor):
    return Tensor(
        value=f.relu(x.value),
        prevs=[Edge(prev=x, local_grad=f.relu_prime(x.value), backward_fn=_pointwise)],
    )


def tanh(x: Tensor):
    return Tensor(
        value=f.tanh(x.value),
        prevs=[Edge(prev=x, local_grad=f.tanh_prime(x.value), backward_fn=_pointwise)],
    )
