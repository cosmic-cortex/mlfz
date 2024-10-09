import numpy as np
from collections import namedtuple
from typing import List, Tuple


##########################
# Broadcasting utilities #
##########################


def pad_shapes(x: Tuple, y: Tuple):
    diff = len(y) - len(x)

    if diff > 0:
        x = (1,) * diff + x
    elif diff < 0:
        y = (1,) * -diff + y

    return x, y


def check_shapes(x: Tuple, y: Tuple):
    return all(d1 == d2 or d1 == 1 or d2 == 1 for d1, d2 in zip(x, y))


def broadcasted_shape(x: Tuple, y: Tuple):
    x, y = pad_shapes(x, y)

    if not check_shapes(x, y):
        raise ValueError(f"The shapes {x} and {y} are not broadcastable.")

    return tuple(max(i, j) for i, j in zip(x, y))


def align_tuple(x: Tuple, align_to: Tuple):
    tpl_list = list(x)
    return tuple(
        tpl_list.pop(tpl_list.index(val)) if val in tpl_list else 1 for val in align_to
    )


#################################################
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


def _tile(tensor, local_grad, prev):
    """
    Blows up the backwards gradient to match the backwards gradient
    of the previous tensor.
    """
    backwards_grad_new_shape = align_tuple(tensor.shape, prev.shape)
    backwards_grad = tensor.backwards_grad.reshape(backwards_grad_new_shape)
    return np.broadcast_to(backwards_grad, prev.shape)


def _weighted_tile(tensor, local_grad, prev):
    """
    Broadcasts the backwards gradient to match the backwards gradient
    of the previous tensor, then weighs it with the local gradient.
    """

    return _tile(tensor, local_grad, prev) * local_grad


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

    # padding the shape of the backwards gradient with ones
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
        bs_shape = broadcasted_shape(x.shape, y.shape)
        x, y = x.broadcast_to(bs_shape), y.broadcast_to(bs_shape)
    return x, y


####################
# Tensor functions #
####################


def sum(x, axis=None):
    return x.sum(axis=axis)


def mean(x, axis=None):
    return x.mean(axis=axis)
