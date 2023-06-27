#!/usr/bin/env python3
"""Creates momentum opt operation
for a neural network"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """creates operation and returns it"""

    opt = tf.train.MomentumOptimizer(learning_rate=alpha,
                                     momentum=beta1)
    train_op = opt.minimize(loss)

    return train_op
