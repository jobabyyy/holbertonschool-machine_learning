#!/usr/bin/env python3
"""RNNs: Bidirectional 
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ Function that performs forward
        propagation for a bidirectional RNN.
    
    Args:
        bi_cell: is an instance of BidirectinalCell
                 that will be used for the forward propagation
        X: is the data to be used, given as a numpy.ndarray
           of shape (t, m, i)
        t: is the maximum number of time steps
        m: is the batch size
        i: is the dimensionality of the data
        h_0: is the initial hidden state in the forward
             direction, given as a numpy.ndarray of shape (m, h)
        h: is the dimensionality of the hidden state
        h_t: is the initial hidden state in the backward
            direction, given as a numpy.ndarray of shape (m, h)
    Returns: H, Y
             H: is a numpy.ndarray containing all
                of the concatenated hidden states
             Y: is a numpy.ndarray containing all
                of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    
    H_forward[0] = h_0
    H_backward[t-1] = h_t
    
    for step in range(t):
        H_forward[step] = bi_cell.forward(H_forward[step-1],
                                          X[step])
        
    for step in range(t-2, -1, -1):  # start from t-2 instead of t-1
        H_backward[step] = bi_cell.backward(H_backward[step+1],
                                            X[step])

    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)
    
    return H, Y
