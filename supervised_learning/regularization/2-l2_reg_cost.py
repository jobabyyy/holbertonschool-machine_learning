#!/usr/bin/env python3
"""calc the cost of a neural
network w the l2 regularization"""


import tensorflow as tf


def l2_reg_cost(cost):
    """calcs the cost of neural network w
    l2 reg"""
    return (cost + tf.losses.get_regularization_losses())
