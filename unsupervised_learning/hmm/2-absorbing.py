#!/usr/bin/env python3
"""HMM: Absorboing
determines if a markov chain
is absorbing"""


import numpy as np


def absorbing(P):
    """
    P: a square 2D numpy.ndarray of shape (n, n)
      - P[i, j]: probability of transitioning
                 from state i to state j
      - N: is the number of states in the
           markov chain.
    Returns: True if it is absorbing, or
             False on failure."""
    if not isinstance(P, np.ndarray
                      ) or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return False
    # n calcs the num of states
    n = P.shape[0]

    # looking for atleast one absorbing state
    absorbing_states = np.where(np.diag(P) == 1)[0]
    if len(absorbing_states) == 0:
        return False
    # checking for paths
    for i in range(n):
        visited_states = set()  # avoids infinite loops
        stack = [i]  # performs depth-first search
        while stack:
            state = stack.pop()
            if state in visited_states:
                continue
            visited_states.add(state)
            if state in absorbing_states:
                break  # path found
            for j in range(n):
                if P[state, j] > 0:
                    stack.append(j)

        if not any(state in absorbing_states for state in visited_states):
            return False

    return True
