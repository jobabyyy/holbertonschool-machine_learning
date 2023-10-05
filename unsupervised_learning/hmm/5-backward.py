#!/usr/bin/env python3
"""HMM: The Backward Algorithm
performs the backward algorithm
for a hidden markov model.
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Observation: numpy.ndarray of shape (T,) that contains
                 the index of the observation
    T: is the number of observations
    Emission: is a numpy.ndarray of shape (N, M) containing
              the emission probability of a specific
              observation given a hidden state
    Emission[i, j]: is the probability of observing j given
                    the hidden state i
    N: is the number of hidden states
    M: is the number of all possible observations
    Transition: a 2D numpy.ndarray of shape (N, N) containing
                the transition probabilities
    Transition[i, j]: is the probability of transitioning from
                      the hidden state i to j
    Initial: a numpy.ndarray of shape (N, 1) containing the
             probability of starting in a particular hidden state
    Returns: P, B, or None, None on failure
             P: is the likelihood of the observations
                given the model
             B: is a numpy.ndarray of shape (N, T)
                containing the backward path probabilities
             B[i, j]: the probability of generating the
                      future observations from hidden
                      state i at time j
    """
    if (
        not isinstance(Observation, np.ndarray)
        or not isinstance(Emission, np.ndarray)
        or not isinstance(Transition, np.ndarray)
        or not isinstance(Initial, np.ndarray)
    ):
        return None, None

    T = Observation.shape[0]
    N, M = Emission.shape

    if (
        len(Emission.shape) != 2
        or len(Transition.shape) != 2
        or len(Initial.shape) != 2
        or Emission.shape[0] != N
        or Transition.shape[0] != N
        or Transition.shape[1] != N
        or Initial.shape[0] != N
        or Initial.shape[1] != 1
    ):
        return None, None

    if T == 0:
        return None, None

    B = np.zeros((N, T))
    P = 0

    # init the last column of the backward matrix to 1
    B[:, T - 1] = 1

    # calc backward matrix
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(B[j, t + 1] * Transition[i, j] * Emission[
                             j, Observation[t + 1]] for j in range(N))

    # calc likelihood of the observations
    P = np.sum(Initial[i, 0] * Emission[i,
               Observation[0]] * B[i, 0] for i in range(N))

    return P, B
