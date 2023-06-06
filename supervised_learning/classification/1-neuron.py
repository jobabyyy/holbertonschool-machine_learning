#!/usr/bin/env python3
"""Class that defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """def a single neuron"""
    def __init__(self, nx):
        """init self & nx"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0


    def forward_prop(self, X):
        """Calculate forward propagation"""
        Z = np.matmul(self.W, X) + self.b
        self.A = 1 / (1 + np.exp(-Z))
        return self.A
