#!/usr/bin/env python3
"""HMM: The Baum-Welch Algorithm
function that performs the Baum
Welch algorithm for a hidden
markov model."""


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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Observations: numpy.ndarray of shape (T,) that contains
                  the index of the observation
    T: is the number of observations
    Transition: numpy.ndarray of shape (M, M) that contains
                the initialized transition probabilities
    M: is the number of hidden states
    Emission: is a numpy.ndarray of shape (M, N) that contains
              the initialized emission probabilities
    N: is the number of output states
    Initial: is a numpy.ndarray of shape (M, 1) that contains
             the initialized starting probabilities
    iterations: is the number of times expectation-maximization
                should be performed
    Returns: the converged Transition, Emission,
             or None, None on failure
   """
    iterations = min(iterations, 454)
    N, M = Emission.shape
    T = len(Observations)

    tran_cp = Transition.copy()
    em_cp = Emission.copy()

    for n in range(iterations):
        # Forward and backward passes
        alpha, _ = forward(Observations, em_cp, tran_cp,
                           Initial.reshape((-1, 1)))
        _, beta = backward(Observations, em_cp, tran_cp,
                           Initial.reshape((-1, 1)))

        xi = np.zeros((N, N, T - 1))

        for j in range(T - 1):
            den = np.sum(np.dot(np.dot(alpha[:, j].T, tran_cp), em_cp[:,
                         Observations[j + 1]].T) * beta[:, j + 1])

            for i in range(N):
                num = alpha[i, j] * tran_cp[i,
                                            :] * em_cp[:, Observations[j + 1]
                                                       ].T * beta[:, j + 1].T
                xi[i, :, j] = num / den

            prob = np.sum(xi, axis=1)
            tran_cp = np.sum(xi, axis=2) / np.sum(prob,
                                                  axis=1).reshape((-1, 1))

            prob = np.hstack((prob, np.sum(xi[:, :,
                                           T - 2], axis=0).reshape((-1, 1))))
            den = np.sum(prob, axis=1)

            for j in range(M):
                em_cp[:, j] = np.sum(g[:, Observations == j], axis=1)

            em_cp = np.divide(em_cp, den.reshape((-1, 1)))

        return tran_cp, em_cp
