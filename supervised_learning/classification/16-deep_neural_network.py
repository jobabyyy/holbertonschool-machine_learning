#!/usr/bin/env python3
"""Deep Neural Network class"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network"""
    def __init__(self, nx, layers):
        """init self, nx, and layers"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__layer_count = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer in range(1, self.__layer_count + 1):
            self.initialize_weights_for_layer(layer, layers, nx)

    def initialize_weights_for_layer(self, layer, layers, nx):
        """init weight"""
        if layer == 1:
            weight = np.random.randn(layers[layer - 1], nx) * np.sqrt(2 / nx)
        else:
            weight = np.random.randn(layers[layer - 1], layers[layer - 2]) \
                     * np.sqrt(2 / layers[layer - 2])
        self.__weights['W' + str(layer)] = weight
        self.__weights['b' + str(layer)] = np.zeros((layers[layer - 1], 1))

    @property
    def layer_count(self):
        return self.__layer_count

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
