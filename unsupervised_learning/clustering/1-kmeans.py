#!/usr/bin/env python3
"""Clustering: Kmeans."""

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
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape

    # calc min & max values
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)

    # init centroids using MUD
    centroids = np.random.uniform(low=minimum, high=maximum,
                                  size=(k, d))
    return centroids


def kmeans(X, k, iterations=1000):
    """fun kmeans that performs K-means
    on a dataset.
    x: is a numpy.ndarray of shape (n, d).
       n - is the num of data points.
       d - is the num of dimensions
    K: is a positive int containing the max
    num of iterations that should be performed.
    Returns: C, clss, or None, None on failure.
             C - is a numpy.ndarray of shape (n,d)
             clss - is a numpy.ndarray of shape (n,)
             containing the index of the
             cluster in C that each data
             point belongs to."""

    if type(k) is not int or k <= 0:
        return None
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    n, d = X.shape

    # init cluster centroid using MUD
    centroids = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0),
                                  size=(k, d))
    for i in range(iterations):
        centroid_copy = centroids.copy()  # copy to check convergence
        # calc distance between closest centroid
        distance = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        clss = np.argmin(distance, axis=0)  # assigning data points

        for j in range(k):
            if len(X[clss == j]) == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 size=(1, d))
            else:
                centroids[j] = (X[clss == j]).mean(axis=0)
        distance = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))

        clss = np.argmin(distance, axis=0)
        # checking 4 convergence
        if np.all(centroid_copy == centroids):
            return centroids, clss

    return centroids, clss
