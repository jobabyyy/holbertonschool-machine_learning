#!/usr/bin/env python3
"""calc the cost of a neural network
w l2 regularization"""


import numpy as np


def 12_reg_cost(cost, lambtha, weights, L, m):
    """calc cost"""
    reg_term = 0

    for layer in range(1, L + 1):
        weight = weight['W' + str(layer)]
        reg_term += np.sum(np.square(weight))

    12_cost = cost + (lambtha / (2 * m)) * reg_term

    return 12_cosr
