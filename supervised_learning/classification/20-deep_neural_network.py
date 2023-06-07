#!/usr/bin/env python3
"""Deep Neural Network continued"""

import numpy as np


class DeepNeuralNetwork:
    """Defines deep neural network"""
    def __init__(self, nx, layers):
        """Init self, nx and layers"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) <= 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lix, layer_size in enumerate(layers, 1):
            if type(layer_size) is not int:
                raise TypeError("layers must be a list of positive integers")
            w = np.random.randn(layer_size, nx) * np.sqrt(2/nx)
            self.__weights["W{}".format(lix)] = w
            self.__weights["b{}".format(lix)] = np.zeros((layer_size, 1))
            nx = layer_size

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """calc prop of neural num"""
        self.__cache["A0"] = X

        for lay in range(1, self.__L + 1):
            A_prev = self.__cache["A{}".format(lay - 1)]
            Wl = self.__weights["W{}".format(lay)]
            bl = self.__weights["b{}".format(lay)]
            Zl = np.dot(Wl, A_prev) + bl
            Al = 1 / (1 + np.exp(-Zl))
            self.__cache["A{}".format(lay)] = Al

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculate cost of the model using logistic regression"""
        m = np.shape(Y)[1]
        loss_Y = np.multiply(Y, np.log(A))
        loss_NY = np.multiply((1 - Y), np.log(1.0000001 - A))
        total_loss = loss_Y + loss_NY
        cost = -np.sum(total_loss) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network predictions"""
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost
