#!/usr/bin/env python3
"""normal class"""


class Normal:
    """Normal distribution"""
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
        """Calculate the z score"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculate the x value of a given z-score"""
        return z * self.stddev + self.mean
    
    def pdf(self, x):
        """Calculates value of PDF for given x-value"""
        coefficient = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coefficient * (2.7182818285 ** exponent)

    def cdf(self, x):
        """Calculates value of CDF for given x-value"""
        z = self.z_score(x)
        return 0.5 * (1 + self.errorf_approx(z / 2 ** 0.5))

    def errorf_approx(self, x):
        """Approximates error function"""
        pi = 3.1415926536
        return (2/(pi**.5))*(x-(x**3)/3 + (x**5)/10 - (x**7)/42 + (x**9)/216)
