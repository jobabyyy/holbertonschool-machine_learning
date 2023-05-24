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
            mean = sum(data) / len(data)
            vary = sum([(result - mean) ** 2 for result in data]) / len(data)
            self.p = 1 - (vary / mean)
            self.n = round((sum(data) / self.p) / len(data))
            self.p = float(mean / self.n)

    def factorial(self, k):
        """Calculates the factorial of k."""
        fact = 1
        for i in range(1, int(k) + 1):
            fact *= i
        return fact

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes."""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            binomial_coefficient = (
                self.factorial(self.n) /
                (self.factorial(k) * self.factorial(self.n - k))
            )
            return (
                binomial_coefficient *
                (self.p ** k) *
                ((1 - self.p) ** (self.n - k))
            )
    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes."""
        k = int(k)
        if k < 0:
            return 0
        else:
            cumulative = 0
            for i in range(k + 1):
                cumulative += self.pmf(i)
            return cumulative
