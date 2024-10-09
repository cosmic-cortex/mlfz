import numpy as np
from collections import namedtuple
from typing import List

from .utils import (
    _transpose,
    _weighted_tile,
    _reduce,
    _pointwise,
    _matmul_right,
    _matmul_left,
    _reshape,
    precast,
)


Edge = namedtuple(
    "Edge", ["prev", "local_grad", "backward_fn"], defaults=[None, None, None]
)


class Tensor:
    # the __array_priority__ makes sure that when used in combination
    # with NumPy arrays, the Tensor operations take precedence
    __array_priority__ = 1.0

    def __init__(
        self,
        value: np.ndarray,
        prevs: List = None,
    ):
        self.value = np.array(value)
        self.prevs = prevs if prevs is not None else []
        self.backwards_grad = np.zeros_like(value)

    def __repr__(self):
        return self.value.__repr__().replace("array", "Tensor")

    def __len__(self):
        return len(self.value)

    def _backward_step(self):
        for prev, local_grad, backward_fn in self.prevs:
            prev.backwards_grad += backward_fn(self, local_grad, prev)

    def _get_graph(self, zero_grad=False):
        """
        Returns the Tensor nodes as a topologically ordered list, ending with self.

        Args:
            zero_grad: Resets the backwards_grad attribute for all nodes if true.

        Source: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py#L54
        """
        ordered_tensors = []
        visited_tensors = set()

        def traverse_graph(x):
            if x not in visited_tensors:
                visited_tensors.add(x)

                if zero_grad:
                    x.backwards_grad = 0

                for prev, _, _ in x.prevs:
                    traverse_graph(prev)

                ordered_tensors.append(x)

        traverse_graph(self)

        return ordered_tensors

    def _zero_grad(self):
        self._get_graph(zero_grad=True)

    def gradient_update(self, lr):
        self.value -= lr * self.backwards_grad

    def backward(self, zero_grad=True):
        ordered_tensors = self._get_graph(zero_grad=zero_grad)

        self.backwards_grad = np.ones_like(self.value)

        for tensor in reversed(ordered_tensors):
            tensor._backward_step()

    def __getitem__(self, index):
        return self.value[index]

    def __add__(self, other):
        """
        Pointwise addition of tensors.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        self, other = precast(self, other)

        return Tensor(
            value=self.value + other.value,
            prevs=[
                Edge(prev=self, local_grad=np.ones_like(self), backward_fn=_pointwise),
                Edge(
                    prev=other,
                    local_grad=np.ones_like(other),
                    backward_fn=_pointwise,
                ),
            ],
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return other + self.__neg__()

    def __mul__(self, other):
        """
        Pointwise multiplication of tensors.
        """
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))

        self, other = precast(self, other)

        return Tensor(
            value=self.value * other.value,
            prevs=[
                Edge(prev=self, local_grad=other.value, backward_fn=_pointwise),
                Edge(prev=other, local_grad=self.value, backward_fn=_pointwise),
            ],
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        """
        Pointwise division of tensors.
        """
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))

        self, other = precast(self, other)

        return Tensor(
            value=self.value / other.value,
            prevs=[
                Edge(prev=self, local_grad=1 / self.value, backward_fn=_pointwise),
                Edge(
                    prev=other,
                    local_grad=-other.value / self.value**2,
                    backward_fn=_pointwise,
                ),
            ],
        )

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))

        return other / self

    def __matmul__(self, other):
        """
        Multiplication of tensors.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Tensor(
            value=self.value @ other.value,
            prevs=[
                Edge(prev=self, local_grad=other.value.T, backward_fn=_matmul_left),
                Edge(prev=other, local_grad=self.value.T, backward_fn=_matmul_right),
            ],
        )

    def __pow__(self, exponent):
        """
        Pointwise exponentiation of tensors.
        """
        if not isinstance(exponent, Tensor):
            exponent = Tensor(np.array(exponent))

        self, exponent = precast(self, exponent)

        return Tensor(
            value=self.value**exponent.value,
            prevs=[
                Edge(
                    prev=self,
                    local_grad=exponent.value * (self.value ** (exponent.value - 1)),
                    backward_fn=_pointwise,
                ),
                Edge(
                    prev=exponent,
                    local_grad=np.log(np.abs(self.value))
                    * (self.value**exponent.value),
                    backward_fn=_pointwise,
                ),
            ],
        )

    def __rpow__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))

        return other**self

    @property
    def T(self):
        """
        Transpose of tensors.
        """
        return Tensor(
            value=self.value.T,
            prevs=[
                Edge(
                    prev=self,
                    backward_fn=_transpose,
                )
            ],
        )

    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    def reshape(self, *args):
        return Tensor(
            value=self.value.reshape(*args),
            prevs=[
                Edge(
                    prev=self,
                    backward_fn=_reshape,
                )
            ],
        )

    def sum(self, axis=None, keepdims=False):
        return Tensor(
            value=self.value.sum(axis=axis, keepdims=keepdims),
            prevs=[
                Edge(
                    prev=self,
                    local_grad=1,
                    backward_fn=_weighted_tile,
                )
            ],
        )

    def mean(self, axis=None, keepdims=False):
        N = (
            np.prod(self.shape)
            if axis is None
            else np.prod([self.shape[i] for i in np.atleast_1d(axis)])
        )

        return Tensor(
            value=np.mean(self.value, axis=axis, keepdims=keepdims),
            prevs=[
                Edge(
                    prev=self,
                    local_grad=1 / N,
                    backward_fn=_weighted_tile,
                )
            ],
        )

    def broadcast_to(self, shape):
        return Tensor(
            value=np.broadcast_to(self.value, shape),
            prevs=[
                Edge(
                    prev=self,
                    backward_fn=_reduce,
                )
            ],
        )

    @classmethod
    def ones(cls, *shape):
        return Tensor(value=np.ones(shape))

    @classmethod
    def ones_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Tensor) else tensor
        return Tensor(value=np.ones_like(value))

    @classmethod
    def zeros(cls, *shape):
        return Tensor(value=np.zeros(shape))

    @classmethod
    def zeros_like(cls, tensor):
        value = tensor.value if isinstance(tensor, Tensor) else tensor
        return Tensor(value=np.zeros_like(value))

    @classmethod
    def from_random(cls, *shape, lower=0, upper=1):
        return Tensor(value=np.random.rand(*shape), prevs=[])

    @classmethod
    def random_like(cls, tensor):
        shape = tensor.shape
        return Tensor(value=np.random.rand(*shape), prevs=[])
