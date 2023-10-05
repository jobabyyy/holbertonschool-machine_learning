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
    # steady state probs calculated
    steady_vec = None
     # looking for eigenvec that corresponds to eigenval
    for i in range(len(eigenvals)):
         if np.isclose(eigenvals[i], 1):
             steady_vec = eigenvecs[:, i].real
             break
    if steady_vec is None:
         return None
     # normalize steady state vector
    steady_vec /= np.sum(steady_vec)

    return steady_vec.reshape(1, -1)
