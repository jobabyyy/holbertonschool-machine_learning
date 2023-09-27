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
    cov_matrix = np.cov(X, rowvar=False)

    # Perform eigenvalue decomposition covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigvals)[:-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Select the top 'ndim' eigenvectors
    top_eigvecs = eigvecs[:, :ndim+1]

    # Principal components
    T = np.dot(X, top_eigvecs)

    return T
