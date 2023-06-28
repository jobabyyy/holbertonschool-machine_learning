#!/usr/bin/env python3
"""func to update the learning rate using
inverse time in decay in Numpy"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """updates the learning rate"""

    updated_alpha = alpha / (1 + decay_rate * (global_step // decay_step))

    return updated_alpha
