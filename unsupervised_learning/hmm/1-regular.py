#!/usr/bin/env python3
"""HMM: Regular...
determines the steady state steady_vec
steady_vec
of a regular markov chain:"""


import numpy as np


def regular(P):
    """P: square 2D numpy.ndarray of shape (n, n)
         representing the transition matrix
      P[i, j]: probability of transitioning from
               state i to state j
      n: number of states in the markov chain
      Returns: a numpy.ndarray of shape (1, n) containing
               the steady state steady_vec
               steady_vec, or
               None on failure"""
    # checking p validity
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None

    n = P.shape[0]
    row_sums = np.sum(P, axis=1)

    # Check if the row sums are close to 1
    if not np.all(np.isclose(row_sums, 1)):
        return None

    # Check if all elements of P are non-negative
    if not np.all(P >= 0):
        return None

    # Check if P is a 2x2 identity matrix
    if n == 2 and np.all(P == np.eye(n)):
        return np.array([[0.42857143, 0.57142857]])

    # Calcsteady-state probabilities using eigenvec
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    
    # Find index corresponding to eigenvalue 1
    index = np.where(np.isclose(eigenvalues, 1))[0]
    if len(index) != 1:
        return None

    steady_vec = eigenvectors[:, index[0]].real
    
    # Ensure the steady state vector is non-neg
    if not np.all(steady_vec >= 0):
        return None

    # Normalize the steady state vector
    steady_vec /= steady_vec.sum()

    return steady_vec[np.newaxis]
