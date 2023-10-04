#!/usr/bin/env python3
"""HMM: Viterbi
Calculate the most likely
sequence of hidden states"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Observation: numpy.ndarray of shape (T,) that contains
                 the index of the observation
    T: is the number of observations
    Emission: is a numpy.ndarray of shape (N, M) containing
              the emission probability of a specific
              observation given a hidden state
    Emission[i, j]: is the probability of observing j
                    given the hidden state i
    N: is the number of hidden states
    M: is the number of all possible observations
    Transition: is a 2D numpy.ndarray of shape (N, N) containing
                the transition probabilities
    Transition[i, j]: is the probability of transitioning from
                      the hidden state i to j
    Initial: a numpy.ndarray of shape (N, 1) containing the
             probability of starting in a particular hidden state
    Returns: path, P, or None, None on failure
    path: is the a list of length T containing the most likely
          sequence of hidden states
    P: is the probability of obtaining the path sequence
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

    V = np.zeros((N, T))
    B = np.zeros((N, T), dtype=int)
    path = [0] * T

    # create first column of viterbi
    for i in range(N):
        V[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]
        B[i, 0] = -1

    # now calculating the viterbi matrix && backtracking table
    for t in range(1, T):
        for j in range(N):
            max_prob = 0
            max_state = -1
            for i in range(N):
                prob = V[i, t - 1
                         ] * Transition[i, j] * Emission[j, Observation[t]]
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            V[j, t] = max_prob
            B[j, t] = max_state

    # backtrack to find the most likely path
    path_state = np.argmax(V[:, T - 1])
    path[T - 1] = path_state
    for t in range(T - 2, -1, -1):
        path[t] = B[path[t + 1], t + 1]

    # calculating probability of Path
    P = np.max(V[:, T - 1])

    return path, P
