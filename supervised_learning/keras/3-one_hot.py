#!/usr/bin/env python3
"""func to convert a label vector
into one-hot matrix"""


import numpy as np


def one_hot(labels, classes=None):
    """converts into one hot matrix"""
    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.eye(classes)[labels]

    return one_hot_matrix
