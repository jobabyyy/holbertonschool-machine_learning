#!/usr/bin/env python3
"""Inverse time decay used to create
a learning rate op"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """using inverse time decay"""

    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )
