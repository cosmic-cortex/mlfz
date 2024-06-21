import numpy as np
from collections import namedtuple


Edge = namedtuple("Edge", ["prev", "local_grad"])


class Matrix:
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
        if not isinstance(other, Matrix):
            other = Matrix(other)

        return Matrix(
            value=self.value + other.value,
            prevs=[
                Edge(prev=self, local_grad=np.ones_like(self)),
                Edge(prev=other, local_grad=np.ones_like(other)),
            ],
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return other + self.__neg__()

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return (-1) * self

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        return Matrix(
            value=self.value @ other.value,
            prevs=[
                Edge(prev=self, local_grad=other.value.T),
                Edge(prev=other, local_grad=self.value.T),
            ],
        )

    @property
    def shape(self):
        return self.value.shape

    @classmethod
    def ones_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Matrix) else tensor
        return Matrix(value=np.ones_like(value))

    @classmethod
    def zeros_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Matrix) else tensor
        return Matrix(value=np.zeros_like(value))

    @classmethod
    def from_random(cls, *shape, lower=0, upper=1):
        return Matrix(value=np.random.rand(*shape), prevs=[])
