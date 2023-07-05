#!/usr/bin/env python3
"""updates the weights and biases of a
neural network using gradient
descent with l2 regularization"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases"""
    m = Y.shape[1]

    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
