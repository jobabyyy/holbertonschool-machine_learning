#!/usr/bin/env python3
"""HMM: Markov Chain"""


import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain
    being in a particular state after a
    specified number of iterations:

P: square 2D numpy.ndarray of shape (n, n) representing
   the transition matrix

P[i, j]: the probability of transitioning from
         state i to state j

n: number of states in the markov chain

s: numpy.ndarray of shape (1, n) representing the
   probability of starting in each state

t: number of iterations that the markov chain has been through

Returns: a numpy.ndarray of shape (1, n) representing the
         probability of being in a specific state after t
         iterations, or None on failure"""
    for i in range(t):
        s = np.dot(s, P)

    return s
