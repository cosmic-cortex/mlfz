import math
from mlfz.nn.scalar import Scalar
from mlfz.nn.scalar.functional import sin


x1 = Scalar(1)
x2 = x1 + 3
x3 = 2 * x2
x4 = sin(x3)


def test_forward_pass():
    assert x1.value == 1
    assert x2.value == 1 + 3
    assert x3.value == 2 * (1 + 3)
    assert x4.value == math.sin(2 * (1 + 3))


def test_backward_pass():
    x4.backward()
    assert x4.backwards_grad == 1
    assert x3.backwards_grad == math.cos(2 * (1 + 3))
    assert x2.backwards_grad == 2 * math.cos(2 * (1 + 3))
    assert x1.backwards_grad == 2 * math.cos(2 * (1 + 3))

    # do another backward pass to check if gradient is zeroed
    x4.backward()
    assert x4.backwards_grad == 1
    assert x3.backwards_grad == math.cos(2 * (1 + 3))
    assert x2.backwards_grad == 2 * math.cos(2 * (1 + 3))
    assert x1.backwards_grad == 2 * math.cos(2 * (1 + 3))
