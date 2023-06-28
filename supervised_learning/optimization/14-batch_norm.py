#!/usr/bin/env python3
"""Batch normalization layer
for a neural network in TF"""


import tensorflow as tf


def reate_batch_norm_layer(prev, n, activation):
    epsilon = 1e-8
    
    # Dense layer with kernel initializer
    layer = tf.layers.Dense(units=n, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"), use_bias=False)
    Z = layer(prev)
    
    # Batch normalization
    mean, variance = tf.nn.moments(Z, axes=0)
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, epsilon)
    
    # Activation function
    if activation is not None:
        A = activation(Z_norm)
    else:
        A = Z_norm
    
    return A
