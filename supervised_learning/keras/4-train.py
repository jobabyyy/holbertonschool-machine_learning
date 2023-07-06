#!/usr/bin/env python3
"""trains a model using gradient descent"""


def train_model(network, data, labels,
                batch_size, epochs, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent"""
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, shuffle=shuffle)
    return history
