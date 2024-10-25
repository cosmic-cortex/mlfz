from abc import ABC, abstractmethod


class Model:
    def __call__(self, *args, **kwargs):
        return self.predict(*args, *kwargs)

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_predict(self, X, Y):
        self.fit(X, Y)
        return self.predict(X)
