#!/usr/bin/env python3
"""func to add mat1 & mat2 element-wise"""


def add_matrices2D(mat1, mat2):
    """checking for element length"""
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    rows = len(mat1)
    cols = len(mat1[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            result[i][j] = mat1[i][j] + mat2[i][j]

    return result
