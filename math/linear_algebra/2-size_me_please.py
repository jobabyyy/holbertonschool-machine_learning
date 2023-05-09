#!/usr/bin/env python3
"""print shape of matrix"""


def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))  # Add len of latest dimension 2 shape list
        matrix = matrix[0]  # Move to the 1st element of the current dimension
    return shape
