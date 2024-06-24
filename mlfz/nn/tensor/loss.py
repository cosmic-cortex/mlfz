import math
from .core import Tensor
from .functional import log


def mean_squared_error(preds: Tensor, ys: Tensor) -> Tensor:
    """
    Computes the mean squared error between the predictions and the ground truth.

    Args:
        preds (List[Scalar]): Predictions given by the model.
        ys (List[Scalar]): The ground truth.

    Returns:
        float: The mean squared error between the predictions and the ground truth.
    """
    n_samples = len(preds)

    pass


def binary_cross_entropy(preds: Tensor, ys: Tensor) -> Tensor:
    """
    Computes the binary cross entropy loss between the predictions and the ground truth.

    Args:
        preds (Tensor): Predictions given by the model.
        ys (Tensor): The ground truth.

    Returns:
        float: The binary cross entropy loss between the predictions and the ground truth.
    """

    pass
