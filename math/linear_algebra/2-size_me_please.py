#!/usr/bin/env python3
"""print shape of matrix"""
def matrix_shape(matrix):
    shape = [] 
    while isinstance(matrix, list):
        shape.append(len(matrix))  # Append the length of the current dimension to the shape list
        matrix = matrix[0]  # Move to the first element of the current dimension
    return shape
