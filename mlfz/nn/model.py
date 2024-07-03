from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def gradient_update(self, lr):
        for _, p in self.parameters().items():
            p.gradient_update(lr)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def parameters(self):
        pass

    def parameter_values(self):
        return {name: param.value for name, param in self.parameters().items()}

    def load_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)
