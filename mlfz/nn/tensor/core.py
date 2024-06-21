import numpy as np
from collections import namedtuple


Edge = namedtuple("Edge", ["prev", "local_grad"])


class Tensor:
    def __init__(self, value: np.ndarray, prevs=None):
        self.value = value
        self.prevs = prevs
        self.backwards_grad = np.zeros_like(value)

    def __repr__(self):
        return self.value.__repr__().replace("array", "Tensor")

    def _backward_step(self):
        pass

    def _get_graph(self, zero_grad=False):
        pass

    def _zero_grad(self):
        self._get_graph(zero_grad=True)

    def gradient_update(self):
        self.value -= lr * self.backwards_grad

    def backward(self):
        ordered_scalars = self._get_graph(zero_grad=True)

        self.backwards_grad = 1

        for scalar in reversed(ordered_scalars):
            scalar._backward_step()

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Tensor(
            value=self.value + other.value,
            prevs=[
                Edge(prev=self, local_grad=np.ones_like(self)),
                Edge(self=other, local_grad=np.ones_like(other)),
            ],
        )

    @property
    def shape(self):
        return self.value.shape

    @classmethod
    def ones_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Tensor) else tensor
        return Tensor(value=np.ones_like(value))

    @classmethod
    def zeros_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Tensor) else tensor
        return Tensor(value=np.zeros_like(value))

    @classmethod
    def from_random(cls, *shape, lower=0, upper=1):
        return Tensor(value=np.random.rand(*shape), prevs=[])