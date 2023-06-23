#!/usr/bin/env python3
"""This fun calcs the normalization conts of a matrix"""


import numpy as np


def normalization_constants(X):
    """standardization constants of matrix"""
    # calc mean along each column
    mean = np.mean(X, axis=0)
    # calc standard deviation along each column
    stand = np.std(X, axis=0)

    return mean, stand
