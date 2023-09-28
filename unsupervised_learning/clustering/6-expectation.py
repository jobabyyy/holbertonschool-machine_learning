#!/usr/bin/env python3
"""Clustering: Expectations."""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """function that calculates the
    expectation step in the EM
    algorithm 4 a GMM:
X: is a numpy.ndarray of shape (n, d)
    containing the data set
pi: is a numpy.ndarray of shape (k,)
    containing the priors 4 each cluster
m: is a numpy.ndarray of shape (k, d)
    containing the centroid means 4 each cluster
S: is a numpy.ndarray of shape (k, d, d)
    containing the covariance matrices
    4 each cluste
Returns: g, l, or None, None on failure.
g: is the shape of (k, n)
l: is the likelihood"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None

    k = pi.shape[0]  # num of clusters
    n, d = X.shape  # of data points & dimensions
    g = np.zeros((k, n))  # init posterior possibilies

    # calc possibilities 4 each data point/cluster
    for i in range(k):
        g[i, :] = pi[i] * pdf(X, m[i], S[i])

    # calc total log
    log = np.sum(np.log(np.sum(g, axis=0)))
    # normalize
    g /= np.sum(g, axis=0)

    return g, log
