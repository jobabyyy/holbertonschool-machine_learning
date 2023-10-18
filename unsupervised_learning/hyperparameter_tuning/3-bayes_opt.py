#!/usr/bin/env python3
"""Hyperparameter Tuning:
Initialize Bayesian Optimization w/
GPyOpt. Part 1.
"""


import numpy as np
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
