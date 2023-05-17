#!/usr/bin/env python3
"""calc derivative of polynomial"""


def poly_derivative(poly):
    """poly is a list of coefficients"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])

        if len(derivative) == 0:
            return [0]

    return derivative
