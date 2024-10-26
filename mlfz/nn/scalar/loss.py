import math
from mlfz.nn.scalar.core import Scalar
from mlfz.nn.scalar.functional import log
from typing import List


def mean_squared_error(preds: List[Scalar], ys: List[Scalar]) -> Scalar:
    """
    Computes the mean squared error between the predictions and the ground truth.

    Args:
        preds (List[Scalar]): Predictions given by the model.
        ys (List[Scalar]): The ground truth.

    Returns:
        float: The mean squared error between the predictions and the ground truth.
    """
    return sum([(p - y) ** 2 for p, y in zip(preds, ys)]) / len(preds)


def binary_cross_entropy(preds: List[Scalar], ys: List[Scalar]) -> Scalar:
    """
    Computes the binary cross entropy loss between the predictions and the ground truth.

    Args:
        preds (List[Scalar]): Predictions given by the model.
        ys (List[Scalar]): The ground truth.

    Returns:
        float: The binary cross entropy loss between the predictions and the ground truth.
    """
    epsilon = 1e-16

    return -sum(
        [
            y * log(p + epsilon) + (1 - y) * log(1 - p + epsilon)
            for p, y in zip(preds, ys)
        ]
    ) / len(preds)
