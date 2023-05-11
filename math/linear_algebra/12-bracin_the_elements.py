#!/usr/bin/env python3
"""perfroms mult, division, add"""


def np_elementwise(mat1, mat2):
    """printing results"""
    sum_result = mat1 + mat2
    diff_result = mat1 - mat2
    prod_result = mat1 * mat2
    quotient_result = mat1 / mat2
    return sum_result, diff_result, prod_result, quotient_result
