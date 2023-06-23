#!/usr/bin/env python3
"""This func normalizes a matrix"""


import numpy as np


def normalize(X, m, s):
    """normalize matrix where each
    feauture has 0 mean and unit standard deviation"""

    # This line perfroms the normalization op.
    # 'm' is subtracted from each element in X
    # 'm' is then divided by the result by the ..
    # .. corresponding standard deviation 's'.
    normalized_X = (X - m) / s

    return normalized_X
