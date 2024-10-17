import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return max([0, x])


def relu_prime(x):
    if x < 0:
        return 0
    else:
        return 1


def log_prime(x):
    if x != 0:
        return 1 / math.fabs(x)
    else:
        return 0


def tanh(x):
    return math.tanh(x)


def tanh_prime(x):
    return 1 - math.tanh(x) ** 2
