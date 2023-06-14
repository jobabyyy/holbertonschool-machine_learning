#!/usr/bin/env python3
"""calc the softmax loss of a prediction"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """using args y, y_pred to obtain tensor
    containing the loss of the prediction"""

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred))

    return loss
