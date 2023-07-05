#!/usr/bin/env python3
"""Calc the precision for each class"""


import numpy as np


def precision(confusion):
    """get sum of each value in columns"""
    col_sum = np.sum(confusion, axis=0)

    precision = np.diag(confusion) / col_sum

    return precision
