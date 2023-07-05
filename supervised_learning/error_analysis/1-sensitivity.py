#!/usr/bin/env python3
"""Calculating the sensitivity
for each class in a confusion matrix"""


import numpy as np


def sensitivity(confusion):
    """takes sum of values in each row"""
    row_sum = np.sum(confusion, axis=1)

    sensitivity = np.diag(confusion) / row_sum

    return sensitivity
