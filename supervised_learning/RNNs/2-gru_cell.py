#!/usr/bin/env python3
"""RNNs: GRU Cell
"""

import numpy as np


class GRUCell:
    """
        Class that represents a gated recurrent unit (:
    """
    def __init__(self, i, h, o):
        """
        Class GRUCell that represents a gated recurrent unit.

        Args:
            i: is the dimensionality of the data
            h: is the dimensionality of the hidden state
            o: is the dimensionality of the outputs

        Initializes the public instance attributes
        Wz, Wr, Wh, Wy, bz, br, bh, by.
        The weights should be initialized using a random normal
        distribution.
        The biases should be initialized as zeros.

        - Wz and bz are for the update gate
        - Wr and br are for the reset gate
        - Wh and bh are for the intermediate hidden state
        - Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Args:
            h_prev: is a numpy.ndarray of shape
                    (m, h) containing
                    the previous hidden state
            x_t: is a numpy.ndarray of shape
                 (m, i) that contains
                 the data input for the cell
            m: is the batch size for the data

        Returns: h_next, y
                - h_next: is the next hidden state
                - y: is the output of the cell
        """
        # concat the previous hidden state and input
        concat = np.concatenate((h_prev, x_t), axis=1)
        z_t = np.matmul(concat, self.Wz) + self.bz
        z_t = 1 / (1 + np.exp(-z_t))
        r_t = np.matmul(concat, self.Wr) + self.br
        r_t = 1 / (1 + np.exp(-r_t))

        # concat the previous hidden state and input
        concat2 = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_tilde = np.matmul(concat2, self.Wh) + self.bh
        h_tilde = np.tanh(h_tilde)
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # calc the output and apply softmax activation
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)

        return h_next, y
