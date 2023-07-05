#!/usr/bin/env python3
"""func to calc the specificity for
each class in a confusion matrix"""


import numpy as np


def specificity(confusion):
    """calc sum of all elements"""
    num_classes = confusion.shape[0]
    specificity = np.zeros(num_classes)

    for i in range(num_classes):
        true_negatives = np.sum(confusion) - np.sum(
            confusion[i, :]) - np.sum(confusion[:, i]) + confusion[i, i]
        false_positives = np.sum(confusion[:, i]) - confusion[i, i]
        specificity[i] = true_negatives / (
            true_negatives + false_positives)

    return specificity
