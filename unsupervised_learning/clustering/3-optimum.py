#!/usr/bin/env python3
"""Clustering: Optimum."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """func optimum that tests for the
    optimum number of clusters by variance.
    X: is a numpy.ndarray of shape (n, d)
       containing the data set
    kmin: is a positive integer containing the
          minimum number of clusters to check
          for (inclusive)
    kmax: is a positive integer containing the
          maximum number of clusters to
          check for (inclusive)
    iterations: is a positive integer containing
                the maximum number of iterations
                for K-means
    Returns: results, d_vars, or None, None on failure
             results: is a list containing the outputs of
                      K-means for each cluster size
             d_vars: is a list containing the difference
                     in variance from the smallest cluster size
                     for each cluster size"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None

    n, d = X.shape
    results = []
    d_vars = []

    # Calculate the variance of the smallest cluster (kmin)
    x, clss = kmeans(X, kmin, iterations)
    tiny_guy = variance(X, x)

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None
        var = variance(X, C)  # calc variance of current cluster
        d_var = tiny_guy - var  # calc the diff in variance
        results.append((C, clss))
        d_vars.append(d_var)

    return results, d_vars
