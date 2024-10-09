from mlfz.nn import Model
from mlfz.nn.scalar import Scalar, GradientDescent, mean_squared_error


def test_optimizer():
    a = Scalar(1)
    b = Scalar(1)

    class LinearRegression(Model):
        def __init__(self):
            self.a = a
            self.b = b

        def forward(self, x: Scalar) -> Scalar:
            return self.a * x + self.b

        def parameters(self):
            return {"a": self.a, "b": self.b}

    linear_regressor = LinearRegression()
    optimizer = GradientDescent(model=linear_regressor, loss_fn=mean_squared_error)

    xs = [k * 0.01 for k in range(100)]
    ys = [0.5 * k * 0.01 - 0.5 for k in range(100)]

    optimizer.run(xs, ys)
