#!/usr/bin/env python3
"""func that normalizes an unactive
output of a neural network using
bath normalization"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactive output"""

    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_scaled = gamma * Z_norm + beta

    return Z_scaled
