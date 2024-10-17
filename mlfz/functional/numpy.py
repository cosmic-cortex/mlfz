import numpy as np


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def relu(x: np.ndarray):
    return x * (x > 0)


def relu_prime(x: np.ndarray):
    return (x > 0).astype(int)


def tanh(x: np.ndarray):
    return np.tanh(x)


def tanh_prime(x: np.ndarray):
    return 1 - np.tanh(x) ** 2
