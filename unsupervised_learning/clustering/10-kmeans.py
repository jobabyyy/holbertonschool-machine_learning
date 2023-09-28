#!/usr/bin/env python3
"""Clustering: Kmeans 2"""

import sklearn.cluster


def kmeans(X, k):
    """X is a numpy.ndarray of shape (n, d)
    containing the dataset
k: is the number of clusters
C: is a numpy.ndarray of shape (k, d)
containing the centroid means for each cluster
clss:is a numpy.ndarray of shape (n,)
containing the index of the cluster in C that
each data point belongs to"""

    C, clss, x = sklearn.cluster.kmeans(X, k)

    return C, clss
