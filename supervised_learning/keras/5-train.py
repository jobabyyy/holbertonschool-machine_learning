#!/usr/bin/env python3
"""trains a model using gradient descent
part2 continued..."""

import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, validation_data=None, 
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent
    and analyzes vaildation data"""
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          validation_data=validation_data, verbose=verbose,
                          shuffle=shuffle)
    return history
