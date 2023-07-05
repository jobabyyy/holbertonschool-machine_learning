#!/usr/bin/env python3
"""calc the cost of a neural network
w l2 regularization"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calc cost"""
    reg_term = 0

    for layer in range(1, L + 1):
        weight = weights['W' + str(layer)]
        reg_term += np.sum(np.square(weight))

    l2_cost = cost + (lambtha / (2 * m)) * reg_term

    return l2_cost
