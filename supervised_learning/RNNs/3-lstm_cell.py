#!/usr/bin/env python3
"""RNN: LSTM CELL.
"""


import numpy as np


class LSTMCell():
    """
        Class LSTMCell that represents an LSTM unit. (:
    """
    def __init__(self, i, h, o):
        """ LSTMCell class

        Args:
            i: dimensionality of the dataset
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
                 - Creates the public instance attributes
                   Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo,
                   by that represent the weights
                   and biases of the cell
                Wfand bf: are for the forget gate
                Wuand bu: are for the update gate
                Wcand bc: are for the intermediate cell state
                Woand bo: are for the output gate
                Wyand by: are for the outputs
        """
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros(shape=(1, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros(shape=(1, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros(shape=(1, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        x_t: is a numpy.ndarray of shape (m, i)
             that contains the data input for the cell
        m: is the batche size for the data
        h_prev: is a numpy.ndarray of shape (m, h)
                containing the previous hidden state
        c_prev: is a numpy.ndarray of shape (m, h)
                containing the previous cell state
            - The output of the cell should use a
              softmax activation function
        Returns: h_next, c_next, y
            - h_next is the next hidden state
            - c_next is the next cell state
            - y is the output of the cell
        """
        # concat previous hidden state and input data
        concat = np.concatenate((h_prev, x_t), axis=1)

        # calc the ft and apply sigmoid
        ft = np.matmul(concat, self.Wf) + self.bf
        ft = 1 / (1 + np.exp(-ft))

        # calc the gv and apply sigmoid
        ut = np.matmul(concat, self.Wu) + self.bu
        ut = 1 / (1 + np.exp(-ut))

        # calc the ct and apply sigmoid
        ct = np.matmul(concat, self.Wc) + self.bc
        ct = np.tanh(ct)

        # calc the cell state
        c_next = ft * c_prev + ut * ct

        # calc the output gate values
        ot = np.matmul(concat, self.Wo) + self.bo
        ot = 1 / (1 + np.exp(-ot))

        h_next = ot * np.tanh(c_next)

        yt = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(yt) / np.sum(np.exp(yt), axis=1, keepdims=True)

        return h_next, c_next, y
