from abc import ABC, abstractmethod


class Model:
    def __call__(self, *args, **kwargs):
        return self.predict(*args, *kwargs)

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
