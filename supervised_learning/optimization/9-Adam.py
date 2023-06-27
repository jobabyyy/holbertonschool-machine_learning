#!/usr/bin/env python3
"""Updates a var using Adam Opt"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """new 1st momemnt and new 2nd
    moment to be updated, respectively."""

    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    var -= alpha * (v_corrected / (s_corrected ** 0.5 + epsilon))

    return var, v, s
