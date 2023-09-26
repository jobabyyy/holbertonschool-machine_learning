#!/usr/bin/env python3
"""Dimensionality Reduction:
   Applying PCA on dataset."""


import numpy as np


def pca(X, var=0.95):
    """PCA performance on a dataset using Numpy.
       Returns: the weights matrix 'W'
       which maintains the specified fraction
       of the OG variance."""
    cov_matrix = np.cov(X, rowvar=False)

    # Perform eigenvalue decomposition on the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Determine target variance:
    total_variance = np.sum(eigvals)
    target_variance = var * total_variance

    # Find min num of dimensions
    cumulative_variance = np.cumsum(eigvals)
    num_dimensions = np.argmax(cumulative_variance >= target_variance) + 1

    # Extract principle components
    X = eigvecs[:, :num_dimensions]

    return X
