#!/usr/bin/env python3
"""Class Exponential that reps an exponential distr"""


class Exponential:
    """comment"""
    def __init__(self, data=None, lambtha=1.):
        """initialize an expo distribution"""
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
        """calc the lambtha parameter from given data"""
        num_points = len(data)
        total = sum(data)
        return float(num_points) / total

    def pdf(self, x):
        """calc the value of PDF at given time"""
        if x < 0:
            return 0
        else:
            return self.lambtha * 2.7182818285 ** (-self.lambtha * x)
