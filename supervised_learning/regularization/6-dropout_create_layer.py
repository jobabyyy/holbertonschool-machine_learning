#!/usr/bin/env python3
"""creates a layer of neural network"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=1 - keep_prob)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    dropout_layer = dropout(layer(prev))
    return dropout_layer
