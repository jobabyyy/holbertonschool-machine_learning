#!/usr/bin/env python3
"""func to update a var w momentum opt"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """updates using gradient descent
    returns tuple: updated var at the new moment."""

    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v

    return var, v
