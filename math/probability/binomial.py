#!/usr/bin/env python3
"""class Binomial distribution"""


class Binomial:
    """Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.p = 1 - (
            sum([(x - sum(data) / len(data)) ** 2 for x in data]) /
            (len(data) * sum(data) / len(data)) )

            self.n = round(len(data) / self.p)
            self.p = sum(data) / (self.n * len(data))
