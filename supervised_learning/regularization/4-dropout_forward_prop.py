#!/usr/bin/env python3
"""conducting fwd prop usind Dropout"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """using Dropout to conduct"""
    cache = {'A0': X}
    dropout_masks = {}

    for i in range(1, L + 1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, A_prev) + b

        if i != L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A *= D
            A /= keep_prob
            dropout_masks['D' + str(i)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)

        cache['A' + str(i)] = A

    return cache, dropout_masks
