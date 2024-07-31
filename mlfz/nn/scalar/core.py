import math
import random
from collections import namedtuple
from typing import List


Edge = namedtuple("Edge", ["prev", "local_grad"])


class Scalar:
    """
    A scalar node of the computational graph.

    Attributes:
        value: The scalar value of the node.
        prevs: A list of Edge instances.
        backwards_grad: The value of the gradient, computed during backpropagation.
    """

    def __init__(self, value: float, prevs: List = None):
        self.value = value
        self.prevs = prevs if prevs is not None else []
        self.backwards_grad = 0

    def __repr__(self):
        return f"Scalar({self.value})"

    def _backward_step(self):
        for prev, local_grad in self.prevs:
            prev.backwards_grad += local_grad * self.backwards_grad

    def _get_graph(self, zero_grad=False):
        """
        Returns the Scalar nodes as a topologically ordered list, ending with self.

        Args:
            zero_grad: Resets the backwards_grad attribute for all nodes if true.

        Source: https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py#L54
        """
        ordered_scalars = []
        visited_scalars = set()

        def traverse_graph(x):
            if x not in visited_scalars:
                visited_scalars.add(x)

                if zero_grad:
                    x.backwards_grad = 0

                for prev, _ in x.prevs:
                    traverse_graph(prev)

                ordered_scalars.append(x)

        traverse_graph(self)

        return ordered_scalars

    def _zero_grad(self):
        self._get_graph(zero_grad=True)

    def gradient_update(self, lr):
        self.value -= lr * self.backwards_grad

    def backward(self):
        ordered_scalars = self._get_graph(zero_grad=True)

        self.backwards_grad = 1

        for scalar in reversed(ordered_scalars):
            scalar._backward_step()

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        return Scalar(
            value=self.value + other.value,
            prevs=[Edge(self, 1), Edge(other, 1)],
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        return Scalar(
            value=self.value * other.value,
            prevs=[Edge(self, other.value), Edge(other, self.value)],
        )

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        return Scalar(
            value=self.value / other.value,
            prevs=[
                Edge(self, 1 / other.value),
                Edge(other, -self.value / other.value**2),
            ],
        )

    def __rtruediv__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        return other / self

    def __pow__(self, exponent):
        if not isinstance(exponent, Scalar):
            exponent = Scalar(exponent)

        return Scalar(
            value=self.value**exponent.value,
            prevs=[
                Edge(self, exponent.value * self.value ** (exponent.value - 1)),
                Edge(
                    exponent,
                    math.log(abs(self.value)) * (self.value**exponent.value),
                ),
            ],
        )

    @classmethod
    def from_random(cls, lower=0, upper=1):
        return Scalar(random.uniform(lower, upper))
