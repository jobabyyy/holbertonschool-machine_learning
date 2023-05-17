#!/usr/bin/env python3
"""calc sigma in a func"""


def summation_i_squared(n):
    """sigma"""
    if not isinstance(n, int) or n < 1:
        return None
    else:
        return n*(n+1)*(2*n+1)//6  # // is used to return int.
