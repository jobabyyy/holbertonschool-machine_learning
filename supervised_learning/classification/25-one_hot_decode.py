#!/usr/bin/env python3
"""converts a 1 hot matrix into a vector of labels"""

import numpy as np


def one_hot_decode(one_hot):
    """finding the index"""
    try:
        labels = np.argmax(one_hot, axis=0)
        return labels
    except Exception:
        return None
