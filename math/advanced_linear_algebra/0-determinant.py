#!/usr/bin/env python3

"""A.L.A: Determinant"""

def determinant(matrix):
    """Function to calculate the determinant of a matrix"""
    # Check if the input is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square
    num_rows = len(matrix)
    if num_rows == 0:
        return 1  # The determinant of a 0x0 matrix is 1 by convention
    num_columns = len(matrix[0])
    if num_rows != num_columns:
        raise ValueError("matrix must be a square matrix")

    # Base case for a 1x1 matrix
    if num_rows == 1:
        return matrix[0][0]

    # Base case for a 2x2 matrix
    if num_rows == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Calculate the determinant using recursion for larger matrices
    det = 0
    for col in range(num_columns):
        minor_matrix = [row[:col] + row[col + 1:] for row in matrix[1:]]
        cofactor = matrix[0][col] * determinant(minor_matrix)
        det += cofactor

    return det
