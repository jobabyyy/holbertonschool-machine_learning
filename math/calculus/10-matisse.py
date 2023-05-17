#!/usr/bin/env python3
"""calc derivative of polynomial"""


def poly_derivative(poly):
    # check if poly is a list
    if not isinstance(poly, list):
        return None

    # check if all elements in poly are numbers
    if not all(isinstance(coef, (int, float)) for coef in poly):
        return None

    # if poly represents a constant function, its derivative is 0
    if len(poly) == 1:
        return [0]

    # calculate the derivative
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)

    return derivative
