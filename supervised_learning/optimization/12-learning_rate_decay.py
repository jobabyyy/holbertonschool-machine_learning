#!/usr/bin/env python3
"""Inverse time decay used to create
a learning rate op"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """using inverse time decay"""

    learning_rate = tf.train.inverse_time_decay(
        leaning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    return learning_rate
