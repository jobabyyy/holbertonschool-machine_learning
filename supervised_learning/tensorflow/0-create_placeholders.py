#!/usr/bin/env python3
"""tensorflow placeholders"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """placeholders for data and lables"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    return x, y
