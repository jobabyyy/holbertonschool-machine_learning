#!/usr/bin/env python3
"""func to create layer in tensor"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """create layer in neural network.
    init of prev, n & activation"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.dense(prev, n, activation=activation,
                            kernel_initializer=init)

    return layer
