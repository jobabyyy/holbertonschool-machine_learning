#!/usr/bin/env python3
"""Clustering: Initialize."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function def initialize(X, k): that initializes
variables for a Gaussian Mixture Model:
X: is a numpy.ndarray of shape (n, d) containing the data set
k: is a positive integer containing the number of clusters
Returns: pi, m, S, or None, None, None on failure
pi: is a numpy.ndarray of shape (k,) containing the
    priors for each cluster, initialized evenly
m: is a numpy.ndarray of shape (k, d) containing
   the centroid means for each cluster, initialized with K-means
S: is a numpy.ndarray of shape (k, d, d) containing the
   covariance matrices for each cluster, initialized
   as identity matrices
"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    n, d = X.shape

    # init pi
    pi = np.full((k,), 1/k)

    # init m using kmeans
    m, n = kmeans(X, k)

    # init S as identity matrices
    S = np.zeros((k, d, d))
    for i in range(k):
        S[i] = np.identity(d)

    return pi, m, S
