#!/usr/bin/env python3
"""config"""


import tensorflow.keras as K


def save_config(network, filename):
    """Save the configuration in JSON format"""
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """Loads model w/specific config"""
    with open(filename, 'r') as file:
        config = file.read()
    return K.models.model_from_json(config)
