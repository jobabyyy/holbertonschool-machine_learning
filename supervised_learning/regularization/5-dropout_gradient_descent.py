#!/usr/bin/env python3
"""updates the weights of a neural network w
Dropout regularization using
gradient descent"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights of neural network
    with Dropout using gradient descent"""
    m = Y.shape[1]

    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        A_prev = cache['A' + str(i - 1)]
        D = cache['D' + str(i)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A)) * D / keep_prob

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
