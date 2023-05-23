#!/usr/bin/env python3
"""Class to represent a Poisson distribution"""


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
        """calc lambtha"""
        total = sum(data)
        num_points = len(data)
        return float(total) / num_points

    def pmf(self, k):
        """calc val of the PMF"""
        k = int(k)
        if k < 0:
            return 0
        else:
            return self.calculate_pmf(k)

    def calculate_pmf(self, k):
        """calc val of PMF"""
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        pmf = (self.lambtha ** k) * (2.7182818285 ** -self.lambtha) / factorial
        return pmf

    def cdf(self, k):
        """calc CDF for given num of successes"""
        k = int(k)
        if k < 0:
            return 0
        else:
            return self.calculate_cdf(k)

    def calculate_cdf(self, k):
        """method"""
        cdf = 0
        for i in range(k + 1):
            cdf += self.calculate_pmf(i)
        return cdf
