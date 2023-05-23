#!/usr/bin/env python3
"""Class to represent a Poisson distribution"""


class Poisson:
    """expected num of occurences given frame"""
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
        total = sum(data)
        num_points = len(data)
        return float(total) / num_points

    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0
        else:
            return self._calculate_pmf(k)

    def _calculate_pmf(self, k):
        result = 1.0
        for i in range(1, k + 1):
            result *= self.lambtha / i
        result *= 2.71828 ** -self.lambtha
        return result
