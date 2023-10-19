#!/usr/bin/env python3
"""Hyperparameter Tuning:
Bayesian Optimization -
Acquisition... Updated.
"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    f: black box function be optimized.
    X_init: numpy ndarray of shape (t, 1) reps the input
            already sampled with the blackbox function.
    Y_init: numpy ndarray of shape (t, 1) reps the output
            of box function for each input in X_init
    t: is the number of initial samples.
    bounds: a tuple of (min, max) reps the bounds of the space
            in which to look for the optimal point.
    ac_samples: num of samples that should be analyzed during
                acquistion.
    l: is the length of the parameter for the kernal.
    sigma_f: is the standard deviation given to the
              output of the black-box function.
    xsi: is the exploration-exploitation factor for acquisition
            minimize is a bool determining whether optimization
            should be performed for minimization
            (True) or maximization (False).
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Uses the Expected Improvement acquisition function
        Returns: X_next, EI
        X_next: is a numpy.ndarray of shape (1,) representing
               the next best sample point
        EI: is a numpy.ndarray of shape (ac_samples,)
           containing the expected improvement of each
           potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            f_min = np.min(self.gp.Y)
            imp = f_min - mu - self.xsi
        else:
            f_max = np.max(self.gp.Y)
            imp = mu - f_max - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / (sigma + 1e-8)
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        iterations: is the maximum number of iterations to perform
        If the next proposed point is one that has already been
        sampled, optimization should be stopped early
        Returns: X_opt, Y_opt
        X_opt: is a numpy.ndarray of shape (1,)
               representing the optimal point.
        Y_opt is a numpy.ndarray of shape (1,)
              representing the optimal function value.
        """
        X_opt = None
        Y_opt = None

        for i in range(iterations):
            X_next, i = self.acquisition()

            if X_next in self.gp.X.tolist():
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

            if (Y_opt is None) or (Y_next < Y_opt):
                X_opt = X_next
                Y_opt = Y_next

        return X_opt, Y_opt
