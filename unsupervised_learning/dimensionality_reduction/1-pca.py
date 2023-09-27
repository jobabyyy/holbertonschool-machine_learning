#!/usr/bin/env python3
"""Dimensionality Reduction:
   func PCA that performs PCA
   on a dataset- continued"""


import numpy as np


def pca(X, ndim):
    """func pca that performs PCA
    on a dataset, where n:
                  is the num of
                  data points.
                  where d:
                  is the num
                  of dimensions
                  in each point.
                  ndim: is the new
                  dim of X
    Returns: a numpy.ndarray of shape
    (n, ndim) containing the transformed
    version of 'X'. """
    x_c = X - np.mean(X, axis=0)

    cov_matrix = np.cov(X, rowvar=False)

    # Perform eigenvalue decomposition covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1][:ndim]
    t_eigvals = eigvecs[:, sorted_indices]

    T = np.dot(x_c, t_eigvals)

    return T
