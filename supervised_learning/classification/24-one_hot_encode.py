#!/usr/bin/env python3
"""func to cobert numeric label"""

import numpy as np

def one_hot_encode(Y, classes):
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
