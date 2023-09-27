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

def kmeans(X, k, iteration=1000):
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
    if type(k) is not int or k <= 0 or type(
        iteration) is not int or iteration <= 0:
        return None, None

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None

    n, d = X.shape
    centroids = np.random.uniform(np.min(X, axis=0),
                                 np.max(X, axis=0),
                                 size=(k, d))

    for i in range(iteration):
        cluster_sum = np.zeros((k,d))
        cluster_counts = np.zeros(k, dtype=int)

        for j in range(k):
            # Taking data points closest to current clustet
            distance = np.linalg.norm(X[j] - centroids, axis=1)
            c_cluster = np.argmin(distance)
            cluster_sum[c_cluster] += X[j]
            cluster_counts[c_cluster] += 1

        centroids_copy = centroids.copy()

        for j in range(k):
            if cluster_counts[j] == 0:
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                np.max(X, axis=0), size=(1,d))
            else:
                centroids[j] = cluster_sum[j] / cluster_counts[j]
        if np.array_equal(centroids, centroids_copy):
            return centroids, np.argmin(np.linalg.norm(X[:,
            np.newaxis] - centroids, axis=2), axis=1)

        clss = np.argmin(np.linalg.norm(X[:,
        np.newaxis] - centroids, axis=2), axis=1)

    return centroids, clss
