import numpy as np
from collections import namedtuple
from typing import List


#################################################
# Functions to propagate the gradient backwards #
#################################################


def _pointwise(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Accumulation of the backwards gradient via pointwise multiplication.
    """
    return backwards_grad * local_grad


def _matmul_left(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return backwards_grad @ local_grad


def _matmul_right(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return local_grad @ backwards_grad


def _transpose(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Transposing the backwards gradient.
    """
    return backwards_grad.T


def _broadcast_and_multiply(backwards_grad, local_grad):
    """
    Broadcasts the backwards gradient to match the local gradient.
    """

    y_list = list(backwards_grad.shape)
    backwards_grad_new_shape = tuple(
        y_list.pop(y_list.index(val)) if val in y_list else 1
        for val in local_grad.shape
    )
    backwards_grad = backwards_grad.reshape(backwards_grad_new_shape)
    return np.broadcast_to(backwards_grad, local_grad.shape) * local_grad


def _reshape_and_multiply(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Reshapes the backwards gradient to the shape of the local gradient,
    then multiplies them together pointwise,
    """
    return (local_grad * backwards_grad.reshape(local_grad.shape)).reshape(
        local_grad.shape
    )


def _sum_and_multiply(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Sums the backwards gradient along axes to match  the shape of the
    local gradient, then multiplies them together pointwise.
    """

    backwards_grad_shape = backwards_grad.shape
    local_grad_shape = local_grad.shape

    axes_to_sum = [
        i
        for i in range(len(backwards_grad_shape))
        if backwards_grad_shape[i] not in local_grad_shape
        or backwards_grad_shape.count(backwards_grad_shape[i])
        > local_grad_shape.count(backwards_grad_shape[i])
    ]
    result = np.sum(backwards_grad, axis=tuple(axes_to_sum), keepdims=True)
    return (local_grad * result).reshape(local_grad.shape)


#####################
# Utility functions #
#####################


def precast(x, y):
    if x.shape != y.shape:
        try:
            y = y.broadcast_to(x.shape)
        except:
            try:
                x = x.broadcast_to(y.shape)
            except:
                pass

    return x, y


####################
# Tensor functions #
####################


def sum(x, axis=None):
    return x.sum(axis=axis)


def mean(x, axis=None):
    return x.mean(axis=axis)
