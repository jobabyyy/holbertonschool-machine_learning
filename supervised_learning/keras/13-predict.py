#!/usr/bin/env python3
"""func that makes a predicition
using a neural network"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network
    Returns: the prediction for the data"""
    prediction = network.predict(data, verbose=verbose)

    return prediction
