#!/usr/bin/env python3
"""updates the weights and biases of a
neural network using gradient
descent with l2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases"""
    m = Y.shape[1]

    for layer in range[L, 0, -1]:
        A_prev = cache['A' + str(layer - 1)]
        A = cache['A' + str(layer)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        if layer == L:
            dZ = A - Y
        else:
            dZ = np.matmul(W.T, dZ) * (1 - np.power(A, 2))

        dW = np.matmul(dZ, A_prev.T) / m + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
