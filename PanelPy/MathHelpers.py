# Some helper functions for doing math
import numpy as np
from functools import reduce


def xtx(X):
    """
    Given a matrix X, compute X'X
    :type X: np.array
    :param X: A matrix or array
    :return: X'X
    :rtype: np.array
    """

    (U, sigma, Vstar) = np.linalg.svd(X)
    return times(Vstar.T, np.diag(sigma**2), Vstar)


def times(*args):
    """
    Multiply arbitrarily many matrices
    :param args: matrices or np.arrays or similar
    :return: np.array-like
    """
    return reduce(np.dot, list(args))