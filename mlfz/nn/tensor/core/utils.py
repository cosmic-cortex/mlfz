import numpy as np
from collections import namedtuple
from typing import List


######################
# Broadcasting utils #
######################


def left_pad_shape(X_shape, Y_shape):
    """
    Finds the
    """


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


def _reshape(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Reshapes the backwards gradient to the shape of the local gradient.
    """
    return backwards_grad.reshape(local_grad.shape)


def _reduce(backwards_grad: np.ndarray, local_grad: np.ndarray):
    """
    Sums the backwards gradient along axes to match the shape of the
    local gradient.
    """

    backwards_grad_shape = backwards_grad.shape
    local_grad_shape = local_grad.shape

    # pads the shape of local_grad with ones
    n = len(backwards_grad_shape)
    m = len(local_grad_shape)
    if m < n:
        local_grad_shape = tuple(1 for _ in range(n - m)) + local_grad_shape
    elif m > n:
        raise ValueError(
            f"The shapes {backwards_grad_shape} and {local_grad_shape} are not compatible."
        )

    # find the axes to sum along
    axes_to_sum = [
        i
        for i in range(len(backwards_grad_shape))
        if local_grad_shape[i] == 1 and local_grad_shape[i] != backwards_grad_shape[i]
    ]

    return np.sum(backwards_grad, axis=tuple(axes_to_sum), keepdims=True).reshape(
        local_grad.shape
    )


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
