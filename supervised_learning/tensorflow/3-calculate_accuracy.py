#!/usr/bin/env python3
"""calc the accuracy of a prediciton"""

import tensorflow as tf


def calculate_accuray(y, y_pred):
    """accuracy calculated"""
    correct_predictions = tf.equal(tf.argmax(y, axis=1),
                                   tf.argmax(y_pred, axis=1))
    """accuracy calculated"""
    accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                      tf.float32))

    return accuracy
