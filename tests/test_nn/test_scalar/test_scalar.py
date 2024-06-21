import math
from mlfz.nn.scalar import Scalar
from mlfz.nn.scalar.functional import sin, sigmoid, _sigmoid, _sigmoid_prime


def test_operations_with_nonscalars():
    x1 = Scalar(3)
    x2 = 1 + x1
    x3 = x2 + 4
    x4 = 2 * x3
    x5 = x4 * 3
    x6 = -x5
    x7 = x6 - 1
    x8 = 1 - x7
    x9 = x8**2

    assert x2.value == 1 + 3
    assert x3.value == 1 + 3 + 4
    assert x4.value == 2 * (1 + 3 + 4)
    assert x5.value == 2 * (1 + 3 + 4) * 3
    assert x6.value == (-2) * (1 + 3 + 4) * 3
    assert x7.value == (-2) * (1 + 3 + 4) * 3 - 1
    assert x8.value == 1 + 2 * (1 + 3 + 4) * 3 + 1
    assert x9.value == (1 + 2 * (1 + 3 + 4) * 3 + 1) ** 2


def test_chain():
    x1 = Scalar(1)
    x2 = x1 + 3
    x3 = 2 * x2
    x4 = sin(x3)

    # forward pass
    assert x1.value == 1
    assert x2.value == 1 + 3
    assert x3.value == 2 * (1 + 3)
    assert x4.value == math.sin(2 * (1 + 3))

    # backward pass
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


def test_different_path_lengths():
    x = Scalar(2)
    f1 = x**2
    g1 = x**3
    g2 = sin(g1)
    h = f1 + g2

    # forward pass
    assert f1.value == 2**2
    assert g1.value == 2**3
    assert g2.value == math.sin(2**3)
    assert h.value == 2**2 + math.sin(2**3)

    # backward pass
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


def test_linear_regression():
    a = Scalar(3)
    x = Scalar(2)
    b = Scalar(-1)

    a_times_x = a * x
    l = a_times_x + b
    y = sigmoid(l)

    # forward pass
    assert a_times_x.value == 3 * 2
    assert l.value == 3 * 2 - 1
    assert y.value == _sigmoid(3 * 2 - 1)

    # backward pass
    y.backward()
    assert y.backwards_grad == 1
    assert l.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert b.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert a_times_x.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert a.backwards_grad == 2 * _sigmoid_prime(3 * 2 - 1)
    assert x.backwards_grad == 3 * _sigmoid_prime(3 * 2 - 1)

    # do another backward pass to check if gradient is zeroed
    y.backward()
    assert y.backwards_grad == 1
    assert l.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert b.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert a_times_x.backwards_grad == _sigmoid_prime(3 * 2 - 1)
    assert a.backwards_grad == 2 * _sigmoid_prime(3 * 2 - 1)
    assert x.backwards_grad == 3 * _sigmoid_prime(3 * 2 - 1)


def test_mlp():
    x1, x2 = Scalar(0.5), Scalar(-0.1)

    a11, a12, a21, a22 = 1, 2, 3, 4
    b1, b2 = 0.1, 0.2

    f1 = a11 * x1 + a12 * x2
    f2 = a21 * x1 + a22 * x2
    g1, g2 = sigmoid(f1), sigmoid(f2)
    h = b1 * g1 + b2 * g2

    # forward pass
    assert f1.value == 0.5 * 1 - 0.1 * 2
    assert f2.value == 0.5 * 3 - 0.1 * 4
    assert g1.value == _sigmoid(0.5 * 1 - 0.1 * 2)
    assert g2.value == _sigmoid(0.5 * 3 - 0.1 * 4)
    assert h.value == 0.1 * _sigmoid(0.5 * 1 - 0.1 * 2) + 0.2 * _sigmoid(
        0.5 * 3 - 0.1 * 4
    )

    # backward pass
    h.backward()
    assert h.backwards_grad == 1
    assert g1.backwards_grad == 0.1
    assert g2.backwards_grad == 0.2
    assert f1.backwards_grad == 0.1 * _sigmoid_prime(0.5 * 1 - 0.1 * 2)
    assert f2.backwards_grad == 0.2 * _sigmoid_prime(0.5 * 3 - 0.1 * 4)
    assert x1.backwards_grad == 1 * f1.backwards_grad + 3 * f2.backwards_grad
    assert x2.backwards_grad == 2 * f1.backwards_grad + 4 * f2.backwards_grad
