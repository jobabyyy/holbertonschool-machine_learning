#!/usr/bin/env python3
"""Clustering: Pdf."""


import numpy as np


def pdf(X, m, S):
    """Calculate the probability density function
    of a Gaussian distribution.
    X: is numpy.ndarray data points of shape (n, d).
    m: is numpy.ndarray mean of the distribution
    of shape (d,).
    S: isnumpy.ndarray covariance of the distribution
    of shape (d, d).
    Returns:
        P (numpy.ndarray): PDF values for each data
        point of shape (n,).
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    if m.shape[0] != S.shape[0]:
        return None
    d = m.shape[0]
    # calc the determinant & inverse
    det = np.linalg.det(S)
    if det <= 0:
        return None
    inv_S = np.linalg.inv(S)

    # calc the difference between X and the mean m
    diff = X - m

    # Calculate the PDF values 4 each data point
    exponent = -0.5 * np.sum(np.dot(diff, inv_S) * diff, axis=1)
    P = (1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))) * np.exp(exponent)

    # set minimum
    P[P < 1e-300] = 1e-300

    return P
