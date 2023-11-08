#!/usr/bin/env python3
"""RNNs: Forward Prop"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    We are creating a function that
    performs forward propagation for
    a simple RNN:

    Args:
        rnn_cell: is an instance of RNNCell
                  that will be used for the forward
                  propagation
        X: is the data to be used, given as
           a numpy.ndarray of shape (t, m, i)
        t: is the maximum number of time steps
        m: is the batch size
        i: is the dimensionality of the data
        h_0: is the initial hidden state, given
             as a numpy.ndarray of shape (m, h)
        h: is the dimensionality of the hidden state
        Returns:
                 H, Y
                 H - is a numpy.ndarray containing all
                   of the hidden states
                 Y - is a numpy.ndarray containing all
                   of the outputs
    """
    # dimensions from input data
    time_steps, m, x = X.shape
    hidden_state = h_0.shape[1]
    output = rnn_cell.Wy.shape[1]

    # init arrays to store hidden states & outputs
    H_stored = np.zeros((time_steps, m, hidden_state))
    Y_stored = np.zeros((time_steps, m, output))

    # intit current hidden state
    current_state = h_0

    for step in range(time_steps):
        # fwd prop init for current time step
        current_time = X[step, :]
        current_state, y_t = rnn_cell.forward(current_state,
                                              current_time)

        # store the hidden state and the output
        H_stored[step, :, :] = current_state
        Y_stored[step, :, :] = y_t

    return H_stored, Y_stored
