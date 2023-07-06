#!/usr/bin/env python3
"""Save and Load Model"""


import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire model
    network: the model to save
    filename: the path of the file to save the model to"""
    network.save(filename)


def load_model(filename):
    """Loads an entire model
    filename: the path of the file to load the model from
    Returns: the loaded model"""
    return K.models.load_model(filename)
