from mlfz.nn.scalar import Scalar
from mlfz.nn.scalar.functional import sigmoid, _sigmoid, _sigmoid_prime


a = Scalar(3)
x = Scalar(2)
b = Scalar(-1)

a_times_x = a * x
l = a_times_x + b
y = sigmoid(l)


def test_forward_pass():
    assert a_times_x.value == 3 * 2
    assert l.value == 3 * 2 - 1
    assert y.value == _sigmoid(3 * 2 - 1)


def test_backward_pass():
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
