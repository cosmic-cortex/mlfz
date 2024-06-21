from mlfz.nn.scalar import Scalar
from mlfz.nn.scalar.functional import sigmoid, _sigmoid, _sigmoid_prime


x1, x2 = Scalar(0.5), Scalar(-0.1)

a11, a12, a21, a22 = 1, 2, 3, 4
b1, b2 = 0.1, 0.2

f1 = a11 * x1 + a12 * x2
f2 = a21 * x1 + a22 * x2
g1, g2 = sigmoid(f1), sigmoid(f2)
h = b1 * g1 + b2 * g2


def test_forward_pass():
    assert f1.value == 0.5 * 1 - 0.1 * 2
    assert f2.value == 0.5 * 3 - 0.1 * 4
    assert g1.value == _sigmoid(0.5 * 1 - 0.1 * 2)
    assert g2.value == _sigmoid(0.5 * 3 - 0.1 * 4)
    assert h.value == 0.1 * _sigmoid(0.5 * 1 - 0.1 * 2) + 0.2 * _sigmoid(
        0.5 * 3 - 0.1 * 4
    )


def test_backward_pass():
    h.backward()
    assert h.backwards_grad == 1
    assert g1.backwards_grad == 0.1
    assert g2.backwards_grad == 0.2
    assert f1.backwards_grad == 0.1 * _sigmoid_prime(0.5 * 1 - 0.1 * 2)
    assert f2.backwards_grad == 0.2 * _sigmoid_prime(0.5 * 3 - 0.1 * 4)
    assert x1.backwards_grad == 1 * f1.backwards_grad + 3 * f2.backwards_grad
    assert x2.backwards_grad == 2 * f1.backwards_grad + 4 * f2.backwards_grad
