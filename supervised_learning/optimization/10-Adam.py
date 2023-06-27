#!/usr/bin/env python3
"""Create the training op for a Neural Network
in tensorflow using the Adam op"""


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creating the training op"""

    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
