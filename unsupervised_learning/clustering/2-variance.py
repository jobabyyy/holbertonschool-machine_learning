#!/usr/bin/env python3
"""Clustering: Variance."""

import numpy as np


def variance(X, C):
    """Func variance that calculates
    the total intra-cluster variance
    for a dataset.
    X: is a numpy.ndarray of shape (n, d)
       containing the data set
    C: is a numpy.ndarray of shaoe (k, d)
       containing the centroid means for
       each cluster.
    Returns: var, or None on failure.
    (var is total variance)"""
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None, None
    if X.shape[1] != C.shape[1]:
        return None, None
    
    # calculate distance
    distance = np.sum((X[:, np.newaxis] - C) ** 2, axis=2)

    # get index of centroid
    indices_centroid = np.argmin(distance, axis=1)

    # calculate sum of intra cluster variance
    var = np.sum(distance[np.arange(len(X)), indices_centroid])

    return var
