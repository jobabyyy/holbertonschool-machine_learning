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

    # checking row p sum to aprox 1
    row = np.sum(P, axis=1)
    if not np.all(np.isclose(row, 1)):
        return None
    # checking that all elements are neg
    if not np.all(P >= 0):
        return None

    # calc steady-state probs
    eigenvals, eigenvecs = np.linalg.eig(P.T)

    steady_vec = eigenvecs / eigenvals.sum()
    steady_vec = steady_vec.real
    results = [i.reshape(1, n) for i in np.dot(steady_vec.T,
               P) if (i >= 0).all() and np.isclose(i.sum(), 1)]


    return np.zeros((1, n))
