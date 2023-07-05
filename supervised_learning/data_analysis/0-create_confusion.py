#!/usr/bin/env python3
"""Function to create confusion
matrix"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Multiplication performance between
    transpose of labels
    and logits"""

    confusion = np.matmul(labels.T, logits)

    return confusion
