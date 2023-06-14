#!/usr/bin/env python3
"""creating training op for network"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """trains network using gradient descent"""
    op = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = op.minimize(loss)

    return train_op
