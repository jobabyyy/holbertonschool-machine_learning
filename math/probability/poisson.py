#!/usr/bin/env python3
"""class to rep a poisson distribution"""


class Poisson:
    """expecting certain num of occurences"""
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
        """calculating lambtha"""
        total = sum(data)
        num_points = len(data)
        return float(total) / num_points
