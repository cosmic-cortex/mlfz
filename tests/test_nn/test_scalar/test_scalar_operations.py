from mlfz.nn.scalar import Scalar


x1 = Scalar(3)
x2 = 1 + x1
x3 = x2 + 4
x4 = 2 * x3
x5 = x4 * 3
x6 = -x5
x7 = x6 - 1
x8 = 1 - x7


def test_operations_with_nonscalars():
    assert x2.value == 1 + 3
    assert x3.value == 1 + 3 + 4
    assert x4.value == 2 * (1 + 3 + 4)
    assert x5.value == 2 * (1 + 3 + 4) * 3
    assert x6.value == (-2) * (1 + 3 + 4) * 3
    assert x7.value == (-2) * (1 + 3 + 4) * 3 - 1
    assert x8.value == 1 + 2 * (1 + 3 + 4) * 3 + 1
