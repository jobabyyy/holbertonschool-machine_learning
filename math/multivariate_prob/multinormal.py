#!/usr/bin/env python3
"""class multinormal"""


import numpy as np


class MultiNormal:
    """Multivariate distribution"""
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.zeros((d, d))

        for i in range(d):
            for j in range(i, d):
                self.cov[i, j] = np.mean(
                                         (data[i, :] - self.mean[i,
                                          0]) * (data[j, :] - self.mean[j, 0]))
                self.cov[j, i] = self.cov[i, j]

    def pdf(self, x):
        """func to calc the PDF datapoint"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, _ = self.mean.shape

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        x_minus_mean = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_minus_mean.T,
                                 np.linalg.inv(self.cov)), x_minus_mean)
        coefficient = 1 / ((2 * np.pi) ** (d /
                                           2) * np.sqrt(np.linalg.det
                                                        (self.cov)))

        return coefficient * np.exp(exponent)
