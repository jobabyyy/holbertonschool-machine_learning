#!/usr/bin/env python3
"""A.L.A: Determinant"""


def determinant(matrix):
    """Function to calculate the determinant of a matrix"""
    if not isinstance(matrix, list) or not all(isinstance(
                                               row, list)
                                               for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    num_rows = len(matrix)
    if num_rows == 0:
        raise ValueError("matrix must be a square matrix")
    num_columns = len(matrix[0])
    if num_rows != num_columns:
        raise ValueError("matrix must be a square matrix")
    # Base cases for 0x0 and 1x1 matrices
    if num_rows == 0:
        return 1
    if num_rows == 1:
        return matrix[0][0]
    det = 0

    for col in range(num_columns):
        minor_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(minor_matrix)
        det += cofactor * (-1) ** col

    return det
