#!/usr/bin/env python3
"""class Normal to rep normal distr"""


class Normal:
    """normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize a Normal distribution."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean, self.stddev = self.calculate_mean_stddev(data)

    def calculate_mean_stddev(self, data):
        """Calculate the mean and standard deviation from given data."""
        num_points = len(data)
        mean = sum(data) / num_points
        deviation = [(x - mean) ** 2 for x in data]
        stddev = (sum(deviation) / num_points) ** 0.5
        return mean, stddev

    def z_score(self, x):
        """calc z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calc x value of given z-score"""
        return z * self.stddev + self.mean
