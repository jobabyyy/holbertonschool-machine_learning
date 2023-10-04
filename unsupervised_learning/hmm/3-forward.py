#!/usr/bin/env python3
"""HMM: Forward
Performs the fwd algorithm
for a hidden markov model."""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Observation: numpy.ndarray of shape (T,) that contains
                 the index of the observation
    T: is the number of observations
    Emission: numpy.ndarray of shape (N, M) containing
              the emission probability of a specific observation
              given a hidden state
    Emission[i, j]: probability of observing j given
                    the hidden state i
    N: is the number of hidden states
    M: is the number of all possible observations
    Transition: is a 2D numpy.ndarray of shape (N, N) containing
                the transition probabilities
    Transition[i, j]: the probability of transitioning from
                      the hidden state i to j
    Initial:  a numpy.ndarray of shape (N, 1) containing the
              probability of starting in a particular hidden state
    Returns: P, F, or None, None on failure
    P: the likelihood of the observations given the model
    F: numpy.ndarray of shape (N, T) containing
       the forward path probabilities
    F[i, j]: the probability of being in hidden state
             i at time j given the previous observations
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

    F = np.zeros((N, T))
    P = 0

    # init the fwd probabilities using initial probabilities
    for i in range(N):
        F[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]

    # Calc fwd probabilities for each time step
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[i, t - 1] * Transition[i, j] * Emission[j,
                             Observation[t]] for i in range(N))
    # Calc the likelihood of the observations
    P = np.sum(F[i, T - 1] for i in range(N))

    return P, F
