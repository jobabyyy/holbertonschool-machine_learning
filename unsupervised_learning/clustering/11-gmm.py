#!/usr/bin/env python3
"""Clustering: GMM"""

import sklearn.mixture


def gmm(X, k):
    """X is a numpy.ndarray of shape (n, d)
    containing the dataset
    k: is the number of clusters
    Returns: pi, m, S, clss, bic
    pi: is a numpy.ndarray of shape (k,)
        containing the cluster priors
    m:  is a numpy.ndarray of shape (k, d)
        containing the centroid means
    S:  is a numpy.ndarray of shape (k, d, d)
        containing the covariance matrices
    clss: is a numpy.ndarray of shape (n,)
          containing the cluster indices
          for each data point"""
    gmm_model = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm_model.weights_
    m = gmm_model.means_
    S = gmm_model.covariance_
    clss = gmm_model.predict(X)
    bic = gmm_model.bic(X)

    return pi, m, S, clss, bic
