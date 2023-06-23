#!/usr/bin/env python3
"""This function shuffles data points in 2 matrices"""


import numpy as np


def shuffle_data(X, Y):
    """everyday im shufflin"""

    # retrieves number of data points
    m = X.shape[0]  # X.shape accesses value of 'm'.
    # generates a random permutation of indices
    permutation = np.random.permutation(m)

    # here we are shuffling rows of matrices X and Y
    # using permutation array.
    # rows are arranges in the shufflef order
    # specified by the permutation array.
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y
