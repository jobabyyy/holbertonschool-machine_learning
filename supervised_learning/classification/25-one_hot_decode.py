#!/usr/bin/env python3
"""converts a 1 hot matrix into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """coverts into vector of labels"""
    try:
        labels = np.argmax(one_hot.T, axis=1)
        return labels
    except Exception:
        return None
