#!/usr/bin/env python3
"""Clustering: Initialize."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function def initialize(X, k): that initializes
variables 4 a Gaussian Mixture Model:
X: is a numpy.ndarray of shape (n, d) containing the data set
k: is a positive integer containing the number of clusters
Returns: pi, m, S, or None, None, None on failure
pi: is a numpy.ndarray of shape (k,) containing the
    priors 4 each cluster, initialized evenly
m: is a numpy.ndarray of shape (k, d) containing
   the centroid means 4 each cluster, initialized with K-means
S: is a numpy.ndarray of shape (k, d, d) containing the
   covariance matrices 4 each cluster, initialized
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
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
