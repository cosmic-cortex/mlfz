import math
from mlfz.nn.scalar import Scalar
from mlfz.nn.scalar.functional import sin


x = Scalar(2)
f1 = x**2
g1 = x**3
g2 = sin(g1)
h = f1 + g2


def test_forward_pass():
    assert f1.value == 2**2
    assert g1.value == 2**3
    assert g2.value == math.sin(2**3)
    assert h.value == 2**2 + math.sin(2**3)


def test_backward_pass():
    h.backward()
    assert h.backwards_grad == 1
    assert g2.backwards_grad == 1
    assert g1.backwards_grad == math.cos(2**3)
    assert f1.backwards_grad == 1
    assert x.backwards_grad == (3 * 2**2) * math.cos(2**3) + 2 * 2

    # do another backward pass to check if gradient is zeroed
    h.backward()
    assert h.backwards_grad == 1
    assert g2.backwards_grad == 1
    assert g1.backwards_grad == math.cos(2**3)
    assert f1.backwards_grad == 1
    assert x.backwards_grad == (3 * 2**2) * math.cos(2**3) + 2 * 2
