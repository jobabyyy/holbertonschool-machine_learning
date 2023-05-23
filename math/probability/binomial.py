#!/usr/bin/env python3
"""class Binomial distribution"""


class Binomial:
    """Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Initialize a Binomial distribution."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not 0 < p < 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.n, self.p = self.calculate_n_p(data)

    def calculate_n_p(self, data):
        """Calculate the values of n and p from given data."""
        num_trials = len(data)
        p = sum(data) / num_trials
        n = round(p * num_trials)
        p = n / num_trials
        return int(n), p
