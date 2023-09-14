#!/usr/bin/env python3
"""Bayesian Probabiliy: intersection"""


import numpy as np


def comb(n, k):
    """calculate the binomial coefficient C(n, k)"""
    if 0 <= k <= n:
        return np.math.factorial(n) // (
            np.math.factorial(k) * np.math.factorial(n - k))
    else:
        return 0


def likelihood(x, n, P):
    """func to check likelihood"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    likelihoods = np.array([comb(
                           n, x) * p**x * (1 - p)**(n - x) for p in P])

    # Check if the result is a numpy.ndarray
    if not isinstance(likelihoods, np.ndarray):
        raise TypeError("The result must be a numpy.ndarray")

    return likelihoods


def intersection(x, n, P, Pr):
    """tesing hypotheticals"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x > n:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
                        "Pr must be a numpy.ndarray same shape as P")

    if not np.all((P >= 0) & (P <= 1)):
        raise ValueError(f"All values in P must be in the range [0, 1]")

    if not np.all((Pr >= 0) & (Pr <= 1)):
        raise ValueError(f"All values in Pr must be in the range [0, 1]")

    if not np.isclose(sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    likelihoods = np.array([comb(n,
                            x) * p**x * (1 - p)**(n - x) for p in P])
    posterior = (Pr * likelihoods) / np.sum(Pr * likelihoods)

    return posterior
