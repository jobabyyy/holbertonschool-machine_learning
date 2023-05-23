#!/usr/bin/env python3
"""Class to represent a Poisson distribution"""

import math


class Poisson:
    """Expecting a certain number of occurrences"""

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = self.calculate_lambtha(data)

    def calculate_lambtha(self, data):
        """Calculating lambtha"""
        total = sum(data)
        num_points = len(data)
        return float(total) / num_points

    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0
        else:
            return (
                (self.lambtha ** k) *
                math.exp(-self.lambtha) /
                self.factorial(k)
            )

    @staticmethod
    def factorial(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
