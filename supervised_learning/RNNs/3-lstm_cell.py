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
