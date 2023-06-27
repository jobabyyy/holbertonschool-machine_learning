#!/usr/bin/env python3
"""creates the training op for a neural network"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """RMSProp optimization operation"""

    opt = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                    decay=beta2, epsilon=epsilon)
    train_op = opt.minimize(loss)

    return train_op
