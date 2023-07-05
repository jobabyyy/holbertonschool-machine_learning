#!/usr/bin/env python3
"""calc the cost of a neural
network w the l2 regularization"""


import tensorflow as tf


def l2_reg_cost(cost):
    """calcs the cost of neural network w
    l2 reg"""

    l2_regularizer_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    l2_cost = cost + tf.reduce_sum(l2_regularizer_losses)

    return l2_cost
