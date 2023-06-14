#!/usr/bin/env python3
"""func to create layer in tensor"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """create layer in neural network.
    init of prev, n & activation"""
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init)
    init = tf.contrib.layers.variances_scaling_initializer(mode="FAN_AVG")

    return layer
