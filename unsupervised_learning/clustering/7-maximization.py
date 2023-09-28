#!/usr/bin/env python3
"""Clustering: Max"""


import numpy as np


def maximization(X, g):
    """"X is a numpy.ndarray of shape (n, d)
    containing the data set
g: is a numpy.ndarray of shape (k, n) containing
   the posterior probabilities for each data
   point in each cluster
Returns: pi, m, S, or None, None,
         None on failure
pi: is a numpy.ndarray of shape (k,)
    containing the updated priors for each cluster
m: is a numpy.ndarray of shape (k, d)
   containing the updated centroid means for each cluster
S: is a numpy.ndarray of shape (k, d, d)
   containing the updated covariance matrices for each cluster"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None

    # num of clusters
    k, n = g.shape
    # num of dimensions
    d = X.shape[1]

    # update priors (pi)
    pi = np.sum(g, axis=1) / n

    # update centroid means (m)
    m = np.dot(g, X) / np.sum(g, axis=1, keepdims=True)

    # update covariance matrices (S)
    S = np.zeros((k, d, d))
    for i in range(k):
        diff = X - m[i, :]
        S[i, :, :] = np.dot(g[i, :] * diff.T,
                            diff) / np.sum(g[i, :])

    return pi, m, S
