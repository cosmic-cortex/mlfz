from mlfz.nn.scalar import Scalar
from mlfz.nn import Model


class LinearRegression(Model):
    def __init__(self):
        self.a = 1
        self.b = 1

    def forward(self, x: Scalar) -> Scalar:
        return self.a * x + self.b

    def parameters(self):
        return {"a": self.a, "b": self.b}


def test_load_parameters():
    model = LinearRegression()
    assert model.a == 1
    assert model.b == 1
    model.load_parameters({"a": 42, "b": -13})
    assert model.a == 42
    assert model.b == -13
    assert model.forward(1) == 42 * 1 - 13
