#!/usr/bin/env python3
"""class Binomial distribution"""


class Binomial:
    """Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Assuming that all values in the data are equal to the number of successes,
            # we can find n by taking the maximum value in the data.
            self.n = max(data)
            # And p is the mean of the data divided by n.
            self.p = sum(data) / (self.n * len(data))
