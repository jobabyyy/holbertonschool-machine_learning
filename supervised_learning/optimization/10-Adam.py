#!/usr/bin/env python3
"""Create the training op for a Neural Network
in tensorflow using the Adam op"""


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """creating the training op"""

    opt = tf.train.AdamOptimizer(learning_rate=alpha,
                                 beta1=beta1,
                                 beta2=beta2, epsilon=epsilon)
    train_op = opt.minimize(loss)

    return train_op
