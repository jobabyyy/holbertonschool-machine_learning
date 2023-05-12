#!/usr/bin/env python3
"""still working amongst axis"""


def mat_mul(mat1, mat2):
    """multiplying"""
    rows_mat1 = len(mat1)
    cols_mat1 = len(mat1[0])
    rows_mat2 = len(mat2)
    cols_mat2 = len(mat2[0])

    if cols_mat1 != rows_mat2:
        return None

    result = [[0 for _ in range(cols_mat2)] for _ in range(rows_mat1)]

    for i in range(rows_mat1):
        for j in range(cols_mat2):
            for k in range(cols_mat1):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result
