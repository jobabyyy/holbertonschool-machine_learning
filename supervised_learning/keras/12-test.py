#!/usr/bin/env python3
"""Function that test a neural network"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """tests a neural network using the follwing
    values and returns the loss and accuracy of
    a model"""

    loss = accuracy = network.evaluate(data,
                                       labels, verbose=verbose)

    return [loss, accuracy]
