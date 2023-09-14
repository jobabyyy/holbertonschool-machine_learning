#!/usr/bin/env python3
"""Bayesian Probabiliy: likelihood"""


import numpy as np


def likelihood(x, n, P):
    """func to check likelihood"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0 or x > n:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = np.array([np.math.comb(
                           n, x) * p**x * (1 - p)**(n - x) for p in P])

    # Check if the result is a numpy.ndarray
    if not isinstance(likelihoods, np.ndarray):
        raise TypeError("The result must be a numpy.ndarray")

    return likelihoods
