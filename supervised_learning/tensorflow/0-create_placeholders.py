#!/usr/bin/env python3
"""tensorflow placeholders"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """placeholders for data and lables"""
    x = tf.keras.Input(shape=(nx,), name='x')
    y = tf.keras.Input(shape=(classes,), name='y')

    return x, y
