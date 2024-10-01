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


def _pointwise(backwards_grad, local_grad, prev):
    """
    Accumulation of the backwards gradient via pointwise multiplication.
    """
    return backwards_grad * local_grad


def _matmul_left(backwards_grad, local_grad, prev):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return backwards_grad @ local_grad


def _matmul_right(backwards_grad, local_grad, prev):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return local_grad @ backwards_grad


def _transpose(backwards_grad, local_grad, prev):
    """
    Transposing the backwards gradient.
    """
    return backwards_grad.T


def _broadcast_and_multiply(backwards_grad, local_grad, prev):
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


def _reshape(backwards_grad, local_grad, prev):
    """
    Reshapes the backwards gradient to the shape of the local gradient.
    """
    return backwards_grad.reshape(prev.shape)


def _reduce(backwards_grad, local_grad, prev):
    """
    Sums the backwards gradient along axes to match the shape of the
    local gradient.
    """

    # checking if reduction is possible
    if prev.ndim > backwards_grad.ndim:
        raise ValueError(
            f"Shapes {backwards_grad.shape} and {prev.shape} are not compatible."
        )

    # padding the shape of local_grad with ones
    prev_shape = (1,) * (backwards_grad.ndim - prev.ndim) + prev.shape

    # find the axes to sum along
    axes_to_sum = [
        i
        for i, (bg, lg) in enumerate(zip(backwards_grad.shape, prev_shape))
        if lg == 1 and bg != 1
    ]

    return np.sum(backwards_grad, axis=tuple(axes_to_sum), keepdims=True).reshape(
        prev.shape
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
