from typing import List, Callable
from .core import Scalar
from .loss import mean_squared_error


class GradientDescent:
    def __init__(self, model, loss: Callable):
        self.model = model
        self.loss = loss

    def run(
        self, xs: List[Scalar], ys: List[Scalar], lr: float = 0.01, n_steps: int = 1000
    ):
        for _ in range(n_steps):
            preds = [self.model.forward(x) for x in xs]
            l = self.loss(preds, ys)
            l.backward()
            self.model.gradient_update(lr)
