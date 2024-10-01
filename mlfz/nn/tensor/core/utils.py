import numpy as np
from collections import namedtuple
from typing import List


def blow_up_tuple(tpl, align_to):
    tpl_list = list(tpl)
    return tuple(
        tpl_list.pop(tpl_list.index(val)) if val in tpl_list else 1 for val in align_to
    )


################################################
# Functions to propagate the gradient backwards #
#################################################


def _pointwise(tensor, local_grad, prev):
    """
    Accumulation of the backwards gradient via pointwise multiplication.
    """
    return tensor.backwards_grad * local_grad


def _matmul_left(tensor, local_grad, prev):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return tensor.backwards_grad @ local_grad


def _matmul_right(tensor, local_grad, prev):
    """
    Accumulation of the backwards gradient via matrix multiplication.
    """
    return local_grad @ tensor.backwards_grad


def _transpose(tensor, local_grad, prev):
    """
    Transposing the backwards gradient.
    """
    return tensor.backwards_grad.T


def _broadcast_and_multiply(tensor, local_grad, prev):
    """
    Broadcasts the backwards gradient to match the local gradient.
    """

    backwards_grad_new_shape = blow_up_tuple(tensor.shape, prev.shape)
    backwards_grad = tensor.backwards_grad.reshape(backwards_grad_new_shape)
    return np.broadcast_to(backwards_grad, local_grad.shape) * local_grad


def _reshape(tensor, local_grad, prev):
    """
    Reshapes the backwards gradient to the shape of the local gradient.
    """
    return tensor.backwards_grad.reshape(prev.shape)


def _reduce(tensor, local_grad, prev):
    """
    Sums the backwards gradient along axes to match the shape of the
    local gradient.
    """

    # checking if reduction is possible
    if prev.ndim > tensor.backwards_grad.ndim:
        raise ValueError(
            f"Shapes {tensor.backwards_grad.shape} and {prev.shape} are not compatible."
        )

    # padding the shape of local_grad with ones
    prev_shape = (1,) * (tensor.backwards_grad.ndim - prev.ndim) + prev.shape

    # find the axes to sum along
    axes_to_sum = [
        i
        for i, (bg, lg) in enumerate(zip(tensor.backwards_grad.shape, prev_shape))
        if lg == 1 and bg != 1
    ]

    return np.sum(
        tensor.backwards_grad, axis=tuple(axes_to_sum), keepdims=True
    ).reshape(prev.shape)


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
