#!/usr/bin/env python3
"""func to create layer in tensor"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """create layer in neural network.
    init of prev, n & activation"""
    init = tf.contrib.layers.variances_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')

    output = layer(prev)

    return output
