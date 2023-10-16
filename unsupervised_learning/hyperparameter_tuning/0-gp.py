#!/usr/bin/env python3
"""Hyperparameter Tuning:
Initializing Gaussian Process.
"""

import numpy as np


class GaussianProcess:
    """
        X_init: numpy.ndarray of shape (t, 1)
                repping the inputs already sampled with the
                black-box function
        Y_init: numpy.ndarray of shape (t, 1)
                repping the outputs of the black-box function
                for each input in X_init
        t: number of initial samples
        l: length parameter for the kernel
        sigma_f: is the standard deviation given to the output
        of the black-box function
        - Sets public instance attributes X, Y, l, and sigma_f
          corresponding to the respective constructor inputs
        - Sets public instance attribute K, representing the current
          covariance kernel matrix for the Gaussian process
        - Public instance method def kernel(self, X1, X2):
          calculates the covariance kernel matrix between 2 matrices:
            - X1: is a numpy.ndarray of shape (m, 1)
            - X2: is a numpy.ndarray of shape (n, 1)
              the kernel should use the Radial Basis Function (RBF)
        Returns: the covariance kernel matrix as a
               numpy.ndarray of shape (m, n).
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Init Gaussian instance"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculating covariance matrix

        X1: 1st input contains shape (m, 1)
        X2: 2nd input contains shape (n, 1)
        Returns: matrix shape of (m, n).
        """
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))

        for i in range(m):
            for j in range(n):
                distance = np.linalg.norm(X1[i] - X2[j])
                K[i, j] = self.sigma_f ** np.exp(-0.5 * self.l ** 2)

        return K
