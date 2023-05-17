#!/usr/bin/env python3
"""func to calc the integral of poly"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""

    if not isinstance(poly, list) or \
       not all(isinstance(coef, (int, float)) for coef in poly) or \
       not isinstance(C, (int, float)):
        return None

    # Start with the integration constant
    integral = [C]

    for i in range(len(poly)):
        new_coef = poly[i] / (i + 1)

        # If new_coef is a whole number, represent it as an integer
        if new_coef.is_integer():
            new_coef = int(new_coef)

        integral.append(new_coef)

    # Make the list as small as possible by removing trailing zeros
    while integral[-1] == 0 and len(integral) > 1:
        integral.pop()

    return integral

