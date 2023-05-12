#!/usr/bin/env python3
"""concat along specific axis"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """on axiss"""
    return np.concatenate((mat1, mat2), axis=axis)
