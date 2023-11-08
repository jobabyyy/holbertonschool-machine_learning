#!/usr/bin/env python3
"""RNNs: Deep RNN.
"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function deep_rnn that performs
    forward propagation for a deep RNN.

    Args:
        rnn_cells: is a list of RNNCell instances of
                   length l that will be used for the
                   forward propagation
        l: is the number of layers
        X: is the data to be used, given as a
           numpy.ndarray of shape (t, m, i)
        t: is the maximum number of time steps
        m: is the batch size
        i: is the dimensionality of the data
        h_0: is the initial hidden state, given as
             a numpy.ndarray of shape (l, m, h)
        h: is the dimensionality of the hidden state

        Returns: H, Y
            - H: is a numpy.ndarray containing
                 all of the hidden states
            - Y: is a numpy.ndarray containing
                 all of the outputs
    """
    l = len(rnn_cells)
    t, m, i = X.shape
    i, x, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))
    H[0] = h_0

    for step in range(t):
        h_prev = X[step]
        for layer in range(l):
            rnn_cell = rnn_cells[layer]
            h_next, y = rnn_cell.forward(H[step, layer],
                                         h_prev)
            H[step + 1, layer] = h_next
            h_prev = h_next
        Y[step] = y

    return H, Y
