#!/usr/bin/env python3
"""func to calc the specificity for
each class in a confusion matrix"""


import numpy as np


def specificity(confusion):
    """calc sum of all elements"""
    total_sum = np.sum(confusion)

    neg_sum = total_sum - np.sum(confusion,
                                 axis=1) - np.sum(confusion,
                                           axis=0) + np.diag(confusion)

    neg_total = np.sum(neg_sum)
    specificity = neg_sum / neg_total

    return specificity
