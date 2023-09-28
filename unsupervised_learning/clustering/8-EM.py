#!/usr/bin/env python3
"""Clustering: EM."""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """doc"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol <= 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None

    n, d = X.shape
    # init GMM parameters
    pi, m, S = initialize(X, k)
    l_prev = 0  # Initialize previous log likelihood

    for i in range(iterations):
        # calculate posterior probabilities and log likelihood
        g, log = expectation(X, pi, m, S)

        # update parameters
        pi, m, S = maximization(X, g)

        if verbose and (i % 10 == 0 or i == iterations - 1):
            # print log likelihood
            print(f"Log Likelihood after {i} iterations: {log:.5f}")

        # checking log likelihood
        if abs(log - l_prev) <= tol:
            break

        l_prev = log  # Update previous log likelihood

    return pi, m, S, g, log
