#!/usr/bin/env python3
"""class multinormal"""


import numpy as np


class MultiNormal:
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
