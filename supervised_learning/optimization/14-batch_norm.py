#!/usr/bin/env python3
"""Batch normalization layer
for a neural network in TF"""


import tensorflow as tf


def reate_batch_norm_layer(prev, n, activation):
    """batch created"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    dense = tf.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(
        Z, mean, var, offset=beta, scale=gamma,
        variance_epsilon=1e-8)
    
    return activation(Z_norm)
