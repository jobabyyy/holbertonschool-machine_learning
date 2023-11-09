#!/usr/bin/env python3
"""RNNs: BidirectionalCell --
Output calculations.
"""


import numpy as np


class BidirectionalCell():
    """ Class BidirectionalCell
    """
    def __init__(self, i, h, o):
        """ Class Bidirectional that
            represents a bidirectional
            cell of an RNN.

        class constructor def __init__(self, i, h, o):
        i: is the dimensionality of the data
        h: is the dimensionality
           of the hidden states
        o: is the dimensionality of the outputs
            - Creates the public instance
              attributes Whf, Whb, Wy, bhf, bhb,
              by that represent the weights
              and biases of the cell
        Whf and bhfare: for the hidden states
            in the forward direction
        Whb and bhbare: for the hidden states
            in the backward direction
        Wy and byare: for the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ forward prop

        x_t: is a numpy.ndarray of shape (m, i)
             that contains the data input for the cell
        m: is the batch size for the data
        h_prev: is a numpy.ndarray of shape (m, h)
                containing the previous hidden state
        Returns: h_next -- the next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """ Function backward that
            calculates the hidden
            state in the backward
            direction for one time-step.

        Args:
        x_t: is a numpy.ndarray of shape (m, i) that contains
             the data input for the cell
        m: is the batch size for the data
        h_next: is a numpy.ndarray of shape (m, h)
                containing the next hidden state
        Returns:
        h_pev, the previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """ public instance method
            output that calculates
            all the outputs for the RNN.

        Args:
            H: is a numpy.ndarray of shape (t, m, 2 * h)
               that contains the concatenated hidden
               states from both directions, excluding
               their initialized states
            t: is the number of time steps
            m: is the batch size for the data
            h: is the dimensionality of the hidden states

        Returns:
            Y, the outputs
        """
        Y = np.dot(H, self.Wy) + self.by
        Y = np.exp(Y) / np.sum(np.exp(Y), axis=2,
                               keepdims=True)   

        return Y
    