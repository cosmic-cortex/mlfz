import numpy as np
from collections import namedtuple


Edge = namedtuple("Edge", ["prev", "local_grad"])


class Tensor:
    # the __array_priority__ makes sure that when used in combination
    # with NumPy arrays, the Tensor operations take precedence
    __array_priority__ = 1.0

    def __init__(self, value: np.ndarray, prevs=None):
        self.value = value
        self.prevs = prevs
        self.backwards_grad = np.zeros_like(value)

    def __repr__(self):
        return self.value.__repr__().replace("array", "Tensor")

    def _backward_step(self):
        for prev, local_grad in self.prevs:
            prev.backwards_grad += local_grad * self.backwards_grad

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

                for prev, _ in x.prevs:
                    traverse_graph(prev)

                ordered_tensors.append(x)

        traverse_graph(self)

        return ordered_tensors

    def _zero_grad(self):
        self._get_graph(zero_grad=True)

    def gradient_update(self):
        self.value -= lr * self.backwards_grad

    def backward(self):
        ordered_scalars = self._get_graph(zero_grad=True)

        self.backwards_grad = 1

        for scalar in reversed(ordered_scalars):
            scalar._backward_step()

    def __getitem__(self, index):
        return self.value[index]

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Tensor(
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
        if not isinstance(other, Tensor):
            other = Tensor(np.array(other))

        return Tensor(
            value=self.value * other.value,
            prevs=[
                Edge(prev=self, local_grad=other.value),
                Edge(prev=other, local_grad=self.value),
            ],
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return (-1) * self

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Tensor(
            value=self.value @ other.value,
            prevs=[
                Edge(prev=self, local_grad=other.value.T),
                Edge(prev=other, local_grad=self.value.T),
            ],
        )

    def T(self):
        return Tensor(
            value=self.value.T,
            prevs=[Edge(prev=self, local_grad=np.ones_like(self.value.T))],
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
