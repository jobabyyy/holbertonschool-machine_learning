#!/usr/bin/env python3
"""calc derivative of polynomial"""


def poly_derivative(poly):
    """poly is a list of coefficients"""
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if len(poly) == 1:
        return [0]
    derivative = [poly[i] * i for i in range(1, len(poly))]
    return derivative
