#!/usr/bin/env python3
"""converts a 1 hot matrix into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """coverts into vector of labels"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot.T, axis=1)
