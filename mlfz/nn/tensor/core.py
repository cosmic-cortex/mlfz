import numpy as np
from collections import namedtuple
from numbers import Number
from typing import List


Edge = namedtuple("Edge", ["prev", "local_grad", "backward_fn"])


def _pointwise(backwards_grad, local_grad):
    """
    Accumulation of the backwards gradient via pointwise multiplication.
    """
    return backwards_grad * local_grad


def _matmul_left(backwards_grad, local_grad):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return np.dot(backwards_grad, local_grad)


def _matmul_right(backwards_grad, local_grad):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return np.dot(local_grad, backwards_grad)


def _transpose(backwards_grad, local_grad):
    """
    Transposing the backwards gradient.
    """
    return backwards_grad.T


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
            prev.backwards_grad += backward_fn(self.backwards_grad, local_grad)

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

    def gradient_update(self):
        self.value -= lr * self.backwards_grad

    def backward(self, zero_grad=True):
        ordered_tensors = self._get_graph(zero_grad=zero_grad)

        self.backwards_grad = 1

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

    def T(self):
        """
        Transpose of tensors.
        """
        return Tensor(
            value=self.value.T,
            prevs=[
                Edge(
                    prev=self,
                    local_grad=np.ones_like(self.value.T),
                    backward_fn=_transpose,
                )
            ],
        )

    @property
    def shape(self):
        return self.value.shape

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
