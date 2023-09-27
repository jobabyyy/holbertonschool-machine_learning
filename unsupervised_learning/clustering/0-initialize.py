#!/usr/bin/env python3
"""Clustering: Initialization."""


import numpy as np


def initialize(X, k):
    """function that initializes
    cluster centroids for K-means:
    X: is a numpy.ndarray of shape (n,d)
    n: is the num of data points
    d: is the num of dimensions for
       each data point.
    Returns: a numpy.ndarray of shape (k,d)
            containing the initialized
            centroids for each cluster,
            or None on failure"""
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape

    # check for additonal clusters
    if k > n:
        return None

    # calc min & max values
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    # init centroids using MUD
    centroids = np.random.uniform(low=minimum, high=maximum,
                                  size=(k, d))
    return centroids
