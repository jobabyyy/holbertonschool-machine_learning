#!/usr/bin/env python3
"""func to convert a label vector
into one-hot matrix"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """converts into a one-hot matrix"""

    return K.utils.to_categorical(labels, classes)
