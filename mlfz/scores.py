import numpy as np


def accuracy(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)
