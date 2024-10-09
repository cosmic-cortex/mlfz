from typing import Callable


class GradientDescent:
    def __init__(self, model, loss_fn: Callable):
        self.model = model
        self.loss_fn = loss_fn

    def run(self, xs, ys, lr: float = 0.01, n_steps: int = 1000):
        for _ in range(n_steps):
            preds = [self.model.forward(x) for x in xs]
            l = self.loss_fn(preds, ys)
            l.backward()
            self.model.gradient_update(lr)
