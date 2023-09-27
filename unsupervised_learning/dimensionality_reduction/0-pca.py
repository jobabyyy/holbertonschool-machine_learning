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
    sorted_indices = np.arange(0, len(eigvals), 1)
    sorted_indices = ([x for _, x in sorted(
                      zip(eigvals, sorted_indices))])[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Determine target variance:
    total_variance = np.sum(eigvals)
    exp_variance = eigvals / total_variance

    # Find min num of dimensions
    cumulative_variance = np.cumsum(exp_variance)
    num_dimensions = np.argmax(cumulative_variance >= exp_variance) + 1

    # Extract principle components
    W = eigvecs[:, :num_dimensions + 1]

    return W
